from ultralytics import YOLO

def main():
    # Завантажуємо модель
    model = YOLO('yolov10m.pt') 

    results = model.train(
        data='data.yaml',
        epochs=100,              # 40 може бути замало для стабілізації mAP
        imgsz=640,               
        batch=16,                # 64 може "з'їсти" всю VRAM на m-моделі, 32 — безпечніше для стабільності
        device=0,                
        workers=12,               # Використовуй потужність Ryzen для підготовки кадрів
        optimizer='AdamW',       # AdamW зазвичай краще показує себе у складних інтер'єрних сценах
        patience=10,             # Early stopping: якщо модель не покращується 10 епох — стоп
        save=True,
        project='indoor_projects',
        name='yolov10m_optimized',
        cos_lr=True,             # Косинусний розклад навчання для кращої збіжності
        augment=True             # Обов'язково для детекції в приміщеннях (світло, тіні)
    )

if __name__ == '__main__':
    main()