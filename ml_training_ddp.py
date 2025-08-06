# Distributed Data Parallel for multiple GPUs
# Run within an env that has torch with CUDA installed and on 4 GPUs with 4 workers
#   python3 ml_training_ddp.py -n 10 -b 128
# to train 10 epochs and batch size = 128

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from argparse import ArgumentParser

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # enable P2P communication
    os.environ['NCCL_P2P_DISABLE'] = "0"
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # allow fp16 computation utitlized optimally
    torch.backends.cudnn.benchmark = True
    # allows TensorFloat-32 for better perforamnce on A100/H100
    torch.backends.cuda.matmul.allow_tf32 = True

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, batchsize, num_epochs):
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.54867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load dataset with DistributedSampler
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # num_workers is number of child processes of each loader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=2, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler, pin_memory=True)
    
    # Load ResNet50 model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 has 100 classes
    model = model.to(device)
    
    # Wrap model in DistributedDataParallel:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler()  # enable mixed precision training
    
    # Training loop
    #num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            # mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        epoch_time = time.time() - start_time
        avg_epoch_time = torch.tensor(epoch_time, device=device)
        dist.all_reduce(avg_epoch_time, op=dist.ReduceOp.SUM)
        avg_epoch_time /= world_size

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Avg Time: {avg_epoch_time:.2f}s")
    
    # Save the trained model
    if rank == 0:
        model_path = "model_resnet50_cifar100.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Rank {rank}, Test Accuracy: {100 * correct / total:.2f}%")
    
    cleanup()

def main(batchsize, num_epochs):
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, batchsize, num_epochs), nprocs=world_size, join=True)

if __name__ == '__main__':
    # default values
    batchsize = 128
    num_epochs = 10
    outputfile = "output.txt"

    parser = ArgumentParser()
    parser.add_argument("-b", "--batch-size", dest="batchsize", default="", help="Batch size")
    parser.add_argument("-o", "--output-file", dest="outputfile", default="", help="Output file")
    parser.add_argument("-n", "--num-epochs", dest="num_epochs", default=num_epochs, help="Number of epochs")
    
    args = parser.parse_args()
    if args.batchsize != "":
        batchsize = int(args.batchsize)
    if args.outputfile != "":
        outputfile = args.outputfile
    if str(args.num_epochs) != "":
        num_epochs = int(args.num_epochs)

    main(batchsize, num_epochs)
   
