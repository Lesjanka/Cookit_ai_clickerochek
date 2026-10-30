#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автокликер YOLO + mss + pynput (Linux).

Зависимости:
    pip install ultralytics mss numpy pynput torch

ВАЖНО (Linux): mss и pynput полноценно работают только под X11 (Xorg).
Под Wayland скриншоты/клики могут не срабатывать.
"""

import time

import numpy as np
import torch
import mss
from ultralytics import YOLO
from pynput import keyboard, mouse

# ============================ НАСТРОЙКИ ============================
MODEL_PATH = "runs/detect/cookie_detecror/weights/best.pt"  # исправлена опечатка!

TARGET_CLASSES  = {"game", "upgrade", "ad"}   # что кликаем
CLASS_PRIORITY  = ["ad", "game", "upgrade"]   # порядок важности классов

CONF_THRESHOLD  = 0.7     # порог уверенности
IMG_SIZE        = 640     # размер инференса (меньше = быстрее)
IOU             = 0.45    # порог NMS
MAX_DET         = 50      # максимум детекций за кадр

CLICKS_PER_TARGET = 10    # кликов по game / ad
CLICK_DELAY       = 0.01  # задержка между кликами, сек
CLICK_ALL_TARGETS = True  # True  — кликнуть все цели из одного кадра
                          # False — только самая важная цель

# --- Специальные настройки класса "upgrade" ---
UPGRADE_CLICKS   = 1      # кликов по одной кнопке upgrade (кнопке хватает одного)
UPGRADE_DELAY    = 0.3    # пауза МЕЖДУ кнопками upgrade (при обходе сверху вниз), сек
UPGRADE_COOLDOWN = 2.0    # кулдаун на класс upgrade ПОСЛЕ клика, сек:
                         # пока активен — upgrade не кликается, game/ad работают

IDLE_DELAY   = 0.2        # пауза, если целей нет
RESCAN_DELAY = 0.05       # пауза после серии кликов
# ====================================================================

running = True


def on_press(key):
    """Остановка по ПРОБЕЛУ."""
    global running
    if key == keyboard.Key.space:
        print("\n[!] Пробел нажат — останавливаюсь...")
        running = False
        return False


def main():
    # --- Устройство инференса ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Инференс на: {device}")

    # --- Модель ---
    model = YOLO(MODEL_PATH)

    target_ids = [i for i, name in model.names.items() if name in TARGET_CLASSES]
    if not target_ids:
        raise RuntimeError(f"В модели нет классов {TARGET_CLASSES}. Доступно: {model.names}")
    print(f"[i] Целевые классы: {target_ids}")

    # --- Слушатель клавиатуры ---
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    mouse_ctrl = mouse.Controller()

    # Время последнего клика по upgrade (для кулдауна).
    # Отрицательное значение -> при старте кулдаун сразу "истёк".
    last_upgrade_click = -UPGRADE_COOLDOWN

    # Приоритет классов + правило сортировки (не зависят от кадра)
    prio = {name: i for i, name in enumerate(CLASS_PRIORITY)}

    def sort_key(t):
        name, conf, cx, cy = t
        p = prio.get(name, 99)
        if name == "upgrade":
            return (p, cy, cx)   # upgrade: СВЕРХУ ВНИЗ (по Y), затем слева направо
        return (p, -conf)        # остальные: по убыванию уверенности

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        def grab_bgr() -> np.ndarray:
            """Скриншот -> BGR ndarray."""
            frame = np.asarray(sct.grab(monitor), dtype=np.uint8)  # (h, w, 4) BGRA
            return np.ascontiguousarray(frame[:, :, :3])           # -> BGR

        # Тёплый старт
        model(np.zeros((monitor["height"], monitor["width"], 3), np.uint8),
              imgsz=IMG_SIZE, device=device, verbose=False)

        print("Скрипт запущен. ПРОБЕЛ — остановка.\n")

        try:
            while running:
                t0 = time.perf_counter()

                # 1. Скриншот
                frame = grab_bgr()

                # 2. Инференс
                results = model(frame,
                                conf=CONF_THRESHOLD,
                                iou=IOU,
                                classes=target_ids,
                                imgsz=IMG_SIZE,
                                max_det=MAX_DET,
                                device=device,
                                half=(device == "cuda"),
                                verbose=False)

                # 3. Сбор целей: (class_name, conf, cx, cy)
                targets = []
                for res in results:
                    det = res.boxes.data.cpu().numpy()
                    for x1, y1, x2, y2, conf, cls_id in det:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        targets.append((model.names[int(cls_id)], float(conf), cx, cy))

                if not targets:
                    time.sleep(IDLE_DELAY)
                    continue

                # 4. Сортировка:
                #    ad -> game -> upgrade,
                #    внутри ad/game — по уверенности,
                #    внутри upgrade — СВЕРХУ ВНИЗ
                targets.sort(key=sort_key)

                # 5. Кулдаун upgrade: пока активен — убираем upgrade из целей,
                #    но game/ad продолжают кликаться
                upgrade_ready = (time.monotonic() - last_upgrade_click) >= UPGRADE_COOLDOWN
                clickable = [t for t in targets if t[0] != "upgrade" or upgrade_ready]

                if not clickable:            # остались только upgrade на кулдауне
                    time.sleep(IDLE_DELAY)
                    continue

                skipped = len(targets) - len(clickable)
                if skipped:
                    print(f"    [i] upgrade на кулдауне — пропущено кнопок: {skipped}")

                to_click = clickable if CLICK_ALL_TARGETS else clickable[:1]
                dt_ms = (time.perf_counter() - t0) * 1000

                # 6. Клики
                for name, conf, cx, cy in to_click:
                    if not running:
                        break

                    # upgrade — одиночный клик, game/ad — серия
                    clicks = UPGRADE_CLICKS if name == "upgrade" else CLICKS_PER_TARGET

                    mouse_ctrl.position = (cx, cy)
                    for _ in range(clicks):
                        if not running:
                            break
                        mouse_ctrl.click(mouse.Button.left)
                        time.sleep(CLICK_DELAY)

                    if name == "upgrade":
                        last_upgrade_click = time.monotonic()
                        time.sleep(UPGRADE_DELAY)  # пауза перед следующей кнопкой (ниже)
                        print(f"[+] upgrade  conf={conf:.2f} @ ({cx}, {cy}) [кулдаун {UPGRADE_COOLDOWN}с]")
                    else:
                        print(f"[+] {name:<8} conf={conf:.2f} @ ({cx}, {cy})")

                print(f"    детект: {dt_ms:.0f} мс | целей: {len(targets)}")
                time.sleep(RESCAN_DELAY)

        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            print("\nСкрипт успешно остановлен.")


if __name__ == "__main__":
    main()





