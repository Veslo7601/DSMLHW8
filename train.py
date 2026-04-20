from ultralytics import YOLO

def main():
    # Завантажуємо модель
    model = YOLO('yolov10m.pt') 

    results = model.train(
        data='data.yaml',
        epochs=100,              
        imgsz=640,               
        batch=16,                
        device=0,                
        workers=12,              
        optimizer='AdamW',       
        patience=10,             
        save=True,
        project='indoor_projects',
        name='yolov10m_optimized',
        cos_lr=True,             
        augment=True             
    )

if __name__ == '__main__':
    main()