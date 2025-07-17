import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MeowVocalDataset
from model import VocalNet_CNN
from loss import compute_loss

def train(args):
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VocalNet_CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = MeowVocalDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    start_epoch = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Optionally load checkpoint
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            meow, instrument = [b.to(device) for b in batch]

            optimizer.zero_grad()
            pred = model(meow, instrument)
            loss, logs = loss_fn(pred_waveform, meow_waveform, instrument_waveform)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Train/Loss_Step", loss.item(), epoch * len(dataloader) + step)

            if (step + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Train/Avg_Loss_Epoch", avg_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.ckpt_dir, f"vocalnet_epoch{epoch+1}.pt"))
            print(f"Checkpoint saved at epoch {epoch+1}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_dataset")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume_ckpt", type=str, default=None)

    args = parser.parse_args()
    train(args)
