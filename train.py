import torch
from torch.utils.data import DataLoader
from conditioned_convtasnet import ConditionedConvTasNet
from Dataset_class import SpeakerExtractionDataset
from losses import si_snr
import os


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # Hyperparameters
    # ------------------------
    BATCH_SIZE = 2
    EPOCHS = 30
    MAX_STEPS_PER_EPOCH = 2000
    LR = 5e-5
    CLIP_NORM = 5.0
    SAVE_PATH = "best_model.pth"

    # ------------------------
    # Dataset
    # ------------------------
    train_set = SpeakerExtractionDataset("dataset_split/train")
    val_set = SpeakerExtractionDataset("dataset_split/val")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )

    # ------------------------
    # Model
    # ------------------------
    model = ConditionedConvTasNet()

    model.convtasnet.to(device)
    model.film_layers.to(device)

    optimizer = torch.optim.Adam(
        list(model.convtasnet.parameters()) +
        list(model.film_layers.parameters()),
        lr=LR
    )

    best_val_loss = float("inf")

    # ------------------------
    # Resume Support
    # ------------------------

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH)
        model.convtasnet.load_state_dict(checkpoint["convtasnet"])
        model.film_layers.load_state_dict(checkpoint["film_layers"])
        print("✔ Loaded previous checkpoint")

    # ------------------------
    # Training Loop
    # ------------------------
    for epoch in range(EPOCHS):

        model.train()
        total_train_loss = 0
        step_count = 0

        for mixture, target, enrollment in train_loader:

            mixture = mixture.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(mixture, enrollment)
            loss = si_snr(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.convtasnet.parameters()) +
                list(model.film_layers.parameters()),
                CLIP_NORM
            )

            optimizer.step()

            total_train_loss += loss.item()
            step_count += 1

            if step_count % 200 == 0:
                print(f"Epoch {epoch+1} | Step {step_count} | Loss {loss.item():.4f}")

            if step_count >= MAX_STEPS_PER_EPOCH:
                break

        avg_train_loss = total_train_loss / step_count

        # ===== VALIDATION =====
        model.eval()
        total_val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for mixture, target, enrollment in val_loader:

                mixture = mixture.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(mixture, enrollment)
                loss = si_snr(output, target)

                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps

        print("\n----------------------------------")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print("----------------------------------\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "convtasnet": model.convtasnet.state_dict(),
                "film_layers": model.film_layers.state_dict()
            }, SAVE_PATH)
            print("✔ Saved Best Model\n")


# Windows multiprocessing safety
if __name__ == "__main__":
    main()