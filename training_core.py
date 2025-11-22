import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import streamlit as st
from preprocessing import DataBowl3Detector
from modelyolo4full import YOLOv8_3D_Connectivity

def calculate_metrics(true_labels, predicted_labels):
    """
    חישוב Precision, Recall ו-F1 Score
    :param true_labels: רשימת תוויות אמת (0 או 1)
    :param predicted_labels: רשימת תחזיות המודל (0 או 1)
    :return: precision, recall, f1
    """
    if len(true_labels) == 0 or len(predicted_labels) == 0:
        return 0, 0, 0

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', zero_division=0
    )
    return precision, recall, f1


def run_training_only_train(
    num_epochs: int = 1,
    learning_rate: float = 0.00005,
    train_folder: str = "/content/drive/My Drive/Luna16/fultrain0708",
    val_folder: str = "/content/drive/My Drive/Luna16/streamitcheck",
):
    """
    אימון המודל YOLOv8_3D_Connectivity מתוך האפליקציה (Streamlit/Colab).

    * רץ על DataBowl3Detector בדיוק כמו במחברת.
    * משתמש ב-train_one_epoch_vit ו-validate_one_epoch_vit.
    * מחזיר history: מילון עם לוסים, דיוקים, IoU ו-Dice לכל אפוק.
    """
    import os, sys
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # לוודא שהסקריפטים זמינים לייבוא
    sys.path.append('/content/drive/My Drive/Luna16/scripts')
    from preprocessing import DataBowl3Detector  # בדיוק כמו במחברת
    #st.write(f"🚀 222התחלת אפוק ")

    # ==== קונפיגורציה כמו במחברת (התאמתי לנתיבים מתוך הפרמטרים) ====
    config = {
        'max_stride': 16, 'stride': 4, 'sizelim': 6, 'sizelim2': 30, 'sizelim3': 60,
        'reso': 1, 'blacklist': [], 'aug_scale': False, 'r_rand_crop': 0.3,
        'augtype': {'flip': True, 'rotate': True, 'swap': True},
        'chanel': 1, 'pad_value': 170, 'crop_size': [128, 128, 128],
        'bound_size': 12, 'th_pos_train': 0.6, 'th_pos_val': 0.6,
        'num_neg': 800, 'th_neg': 0.02,
        'anchors': [10.0, 30.0, 60.0],
        'luna_raw': False,
        'datadir_train': train_folder,
        'datadir_val': val_folder,
    }

    # ==== רשימות קבצים ====
    train_split = sorted(list(set(
        os.path.splitext(file)[0].replace('_cropped', '').replace('_label', '')
        for file in os.listdir(train_folder) if file.endswith('.npy')
    )))
    val_split = sorted(list(set(
        os.path.splitext(file)[0].replace('_cropped', '').replace('_label', '')
        for file in os.listdir(val_folder) if file.endswith('.npy')
    )))

    # ==== Dataset + DataLoader ====
    train_dataset = DataBowl3Detector(train_split, config, phase='train')
    val_dataset   = DataBowl3Detector(val_split,   config, phase='val')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,  num_workers=0)
    val_dataloader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=0)

    # ==== מודל ====
    model = YOLOv8_3D_Connectivity().to(device)

    # ==== Optimizer + Scheduler + Losses ====
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    classification_criterion = torch.nn.BCEWithLogitsLoss()
    bbox_criterion = torch.nn.SmoothL1Loss()

    # ==== טעינת checkpoint קיים אם יש (להמשך אימון) ====
    ckpt_path = "/content/drive/MyDrive/Luna16/modelweight/checkpoint.pt"
    start_epoch = 1
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ Loaded checkpoint from epoch {start_epoch-1}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            start_epoch = 1

    # ==== היסטוריה שנחזיר למסך הגרפים ====
    history = {
        "epoch": [],
        "train_cls_loss": [],
        "train_bbox_loss": [],
        "train_acc": [],
        "val_cls_loss": [],
        "val_bbox_loss": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
    }

    # ==== לולאת אימון ====
    for epoch in range(start_epoch, start_epoch + num_epochs):
        #print(f"\n========== Epoch {epoch}/{start_epoch + num_epochs - 1} ==========")
        #st.write(f"🚀 התחלת אפוק {epoch}/{start_epoch + num_epochs - 1}")

        # 🔹 TRAIN
        train_cls_loss, train_bbox_loss, train_acc, train_iou, train_dice = train_one_epoch_vit(
            model,
            train_dataloader,
            classification_criterion,
            bbox_criterion,
            optimizer,
            device,
            scheduler,
            epoch
        )
        #st.write(f"🚀 התחלת validation ")
        # 🔹 VAL
        val_cls_loss, val_bbox_loss, val_acc, val_iou, val_dice = validate_one_epoch_vit(
            model,
            val_dataloader,
            classification_criterion,
            bbox_criterion,
            device,
            epoch
        )
        #st.write(f"🚀 התחלת history ")
        # 🔹 עדכון history
        history["epoch"].append(epoch)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_bbox_loss"].append(train_bbox_loss)
        history["train_acc"].append(train_acc)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_bbox_loss"].append(val_bbox_loss)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        #st.write(f"🚀  history end ")
        # אפשר לעדכן סקדולר לפי ולידציה
        scheduler.step(val_cls_loss)

    print("✅ Training finished.")
    return history





def train_one_epoch_vit(model, dataloader, classification_criterion, bbox_criterion,
                        optimizer, device, scheduler, epochnum):

    model.train()
    total_classification_loss = 0.0
    total_bbox_loss = 0.0
    total_objective_loss = 0.0
    total_correct = 0
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    total_true_negative = 0

    iou_value_total = 0.0
    imap_value_total = 0.0
    dice_value_total = 0.0

    predicted_labels = []
    true_labels = []
    nodule_numbers_final = 0
    #st.write(f"🚀 התחלת אפוק1")
    for batch_idx, batch in enumerate(dataloader):
        #torch.cuda.empty_cache()

        inputs, labels, bboxes, filenames = batch
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(x=inputs, epochnum=epochnum, labels=labels, bboxes=bboxes, mode="train")
        loss = outputs["loss"]
        logits = outputs["logits"]
        has_nodule_detected = outputs["has_nodule_detected"]
        nodule_numbers = outputs["nodule_number"]
        iou_value = outputs["iou_value"]
        imap_value = outputs["imap_value"]
        dice_value = outputs["dice_value"]

        classification_loss_stat = outputs["classification_loss"].item()
        bbox_loss_stat = outputs["regression_loss"].item()
        objective_loss_stat = outputs["objectness_loss"].item()

        loss.backward()
        optimizer.step()

        # סטטיסטיקות
        iou_value_total  += float(iou_value)
        imap_value_total += float(imap_value)
        dice_value_total += float(dice_value)
        nodule_numbers_final += nodule_numbers

        final_result = has_nodule_detected
        predicted_labels.append(int(final_result.item()))

        true_classes = labels[:, 0, 0]  # הנחה: batch_size=1
        true_labels += true_classes.tolist()
        true_vals = true_classes.view(-1)

        total_classification_loss += classification_loss_stat
        total_bbox_loss          += bbox_loss_stata
        total_objective_loss     += objective_loss_stat

        for pred, true in zip([final_result.item()], true_vals):
            true = int(true.item())
            if pred == 1 and true == 1:
                total_true_positive += 1
                total_correct += 1
            elif pred == 0 and true == 0:
                total_true_negative += 1
                total_correct += 1
            elif pred == 1 and true == 0:
                total_false_positive += 1
            elif pred == 0 and true == 1:
                total_false_negative += 1

    precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)

    avg_classification_loss = total_classification_loss / len(dataloader)
    avg_bbox_loss           = 2 * total_bbox_loss / len(dataloader)
    avg_objective_loss      = total_objective_loss / len(dataloader)

    accuracy = total_correct / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0

    ioumean  = 2 * iou_value_total  / len(dataloader)
    imapmean = 2 * imap_value_total / len(dataloader)
    dicemean = 2 * dice_value_total / len(dataloader)

    scheduler.step(avg_classification_loss)

    # (אופציונלי – אם את רוצה שהפונקציה הזו תמשיך לשמור checkpoint)
    # torch.save({...}, "/content/drive/MyDrive/Luna16/modelweight/checkpoint.pt")

    print(f"\nEpoch train {epochnum} Summary:")
    print(f"Losses train- Classification: {avg_classification_loss:.4f}, BBox: {avg_bbox_loss:.4f}")
    print(f"IOU mean: {ioumean:.4f}, Dice mean: {dicemean:.4f}")
    print(f"Accuracy train: {accuracy:.4f}")

    return avg_classification_loss, avg_bbox_loss, accuracy, ioumean, dicemean
    
    
    

def validate_one_epoch_vit(model, dataloader, classification_criterion, bbox_criterion, device, epochnum):
    model.eval()
    total_classification_loss = 0.0
    total_bbox_loss = 0.0
    total_objective_loss = 0.0
    total_correct = 0
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    total_true_negative = 0

    predicted_labels = []
    true_labels = []
    nodule_numbers_final = 0
    iou_value_total = 0.0
    imap_value_total = 0.0
    dice_value_total = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            torch.cuda.empty_cache()

            inputs, labels, bboxes, filenames = batch
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(x=inputs, epochnum=epochnum, labels=labels, bboxes=bboxes, mode="val")

            logits = outputs["logits"]
            has_nodule_detected = outputs["has_nodule_detected"]
            nodule_numbers = outputs["nodule_number"]
            iou_value = outputs["iou_value"]
            imap_value = outputs["imap_value"]
            dice_value = outputs["dice_value"]

            classification_loss_stat = outputs["classification_loss"].item()
            bbox_loss_stat = outputs["regression_loss"].item()
            objective_loss_stat = outputs["objectness_loss"].item()

            best_pred = outputs["best_pred"][0].detach().cpu().numpy()
            cz, cy, cx, rz, ry, rx = best_pred
            Z = Y = X = 128
            cz_v, cy_v, cx_v = cz * Z, cy * Y, cx * X
            print(f"\n🔹 Center voxel indices: Z={cz_v:.2f}, Y={cy_v:.2f}, X={cx_v:.2f}")

            iou_value_total  += float(iou_value)
            imap_value_total += float(imap_value)
            dice_value_total += float(dice_value)
            total_classification_loss += classification_loss_stat
            total_bbox_loss          += bbox_loss_stat
            total_objective_loss     += objective_loss_stat
            nodule_numbers_final     += nodule_numbers

            pred_label = int(has_nodule_detected.item())
            predicted_labels.append(pred_label)

            true_class = int(labels[:, 0, 0].item())
            true_labels.append(true_class)

            if pred_label == 1 and true_class == 1:
                total_true_positive += 1
                total_correct += 1
            elif pred_label == 0 and true_class == 0:
                total_true_negative += 1
                total_correct += 1
            elif pred_label == 1 and true_class == 0:
                total_false_positive += 1
            elif pred_label == 0 and true_class == 1:
                total_false_negative += 1

    n_batches = len(dataloader)
    avg_classification_loss = total_classification_loss / n_batches
    avg_bbox_loss           = 2 * total_bbox_loss / n_batches
    avg_objective_loss      = total_objective_loss / n_batches

    ioumean  = 2 * iou_value_total  / n_batches
    imapmean = 2 * imap_value_total / n_batches
    dicemean = 2 * dice_value_total / n_batches

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\nValidation Epoch {epochnum} Summary:")
    print(f"Losses val- Classification: {avg_classification_loss:.4f}, BBox: {avg_bbox_loss:.4f}")
    print(f"IOU mean val: {ioumean:.4f}, Dice mean val: {dicemean:.4f}")
    print(f"Accuracy val: {accuracy:.4f}")
    print(classification_report(true_labels, predicted_labels))

    return avg_classification_loss, avg_bbox_loss, accuracy, ioumean, dicemean
