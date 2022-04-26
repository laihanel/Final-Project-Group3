# File Descriptions
* dataset.py: read the video data into dataloader.
    - usage example:
    - data_loader = DataLoader(VideoDataset(dataset='ucf101', split='train', clip_len=16),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    - cite: https://github.com/jfzhang95/pytorch-video-recognition/blob/master/dataloaders/dataset.py
* Model_Definition.py: including two models, VC3D and C3D
* Train.py: data training
* Eval_CNN3D.py: data evaluation
* demo.py: making the video demo
* UCF101_9_Load_Data.py: original dataloader, not used due to the error inside the pytorch.