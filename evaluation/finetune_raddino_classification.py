"""
finetune_radiodino_classification.py
====================================
Professional-grade fine-tuning for Brain Tumor Classification using RaDDINO.
Features: Differential LRs, Medical Augmentations, Early Stopping, and 
Multi-metric reporting (Accuracy, F1, Precision, Recall).

Example Usage:
python finetune_radiodino_classification.py --dataset_dir ./data_mri/Training --epochs 50
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torchvision.transforms import (
    Compose, 
    Normalize, 
    ToTensor, 
    Resize, 
    RandomRotation, 
    RandomHorizontalFlip, 
    CenterCrop
)
import evaluate

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RaDDINO for Brain Tumor Classification.")
    
    # Paths
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to ImageFolder dataset (e.g. Brain Tumor Training set)")
    parser.add_argument("--output_dir", type=str, default="./output/radiodino_classification", help="Where to save results")
    parser.add_argument("--model_name", type=str, default="Snarcy/RadioDino-s16",
                        help="Model ID: timm (Snarcy/RadioDino-s16) or transformers (microsoft/rad-dino)")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the ViT backbone")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for the classification head")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup fraction for scheduler")
    
    # Hardware/Env
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    
    args = parser.parse_args()

    # 1. Setup Data
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_dataset("imagefolder", data_dir=args.dataset_dir)
    
    # 80/20 train/val split
    split_ds = dataset["train"].train_test_split(test_size=0.2, seed=args.seed)
    train_ds = split_ds["train"]
    valid_ds = split_ds["test"]
    
    labels = train_ds.features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    print(f"Found {num_labels} classes: {labels}")

    # 2. Config Processor & Model
    print(f"Loading processor and model: {args.model_name}")

    _is_timm = args.model_name.startswith("Snarcy/") or args.model_name.startswith("hf_hub:")

    if _is_timm:
        import timm
        _hf_id = f"hf_hub:{args.model_name}" if not args.model_name.startswith("hf_hub:") else args.model_name
        _backbone = timm.create_model(_hf_id, pretrained=True, num_classes=num_labels)
        model = _backbone
        size = (224, 224)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Wrap for HuggingFace Trainer compatibility
        import types
        def _fwd(self, pixel_values=None, labels=None, **kw):
            logits = _backbone(pixel_values)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            from transformers.modeling_outputs import ImageClassifierOutput
            return ImageClassifierOutput(loss=loss, logits=logits)
        model.forward = types.MethodType(_fwd, model)
        model.config = type("cfg", (), {"id2label": id2label, "label2id": label2id})()
    else:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        size = (processor.size.get("height", processor.size.get("shortest_edge", 224)),
                processor.size.get("width",  processor.size.get("shortest_edge", 224)))
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

    # Move to device early for optimizer setup
    model.to(args.device)

    # 3. Medical-specific Augmentations

    # MRI scans benefit from slight rotation/flipping as axial/coronal planes can vary
    train_transforms = Compose([
        Resize(size),
        RandomRotation(15),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(img.convert("RGB")) for img in example_batch["image"]]
        return example_batch

    train_ds.set_transform(preprocess_train)
    valid_ds.set_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # 4. Multi-Metric Evaluation
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        prec = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
        rec = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
        
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        }

    # 5. Differential Learning Rates
    # Lower LR for pretrained backbone, higher for the new head
    optimizer_params = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" not in n],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": args.lr_head,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Precision-recall balance is better than pure accuracy
        warmup_steps=100,
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Note: When passing a custom optimizer to Trainer, we must handle the LR scheduler manually 
    # OR let Trainer create it if we provide the optimizer but don't want to handle scheduling.
    # Actually, Trainer accepts the optimizer and will manage the scheduler based on training_args.

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        optimizers=(optimizer, None), # (optimizer, scheduler) - scheduler=None means Trainer creates one
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # 7. Execute Training
    print("Starting training...")
    trainer.train()

    # 8. Save Final Artifacts
    print("Saving best model and backbone...")
    trainer.save_model(args.output_dir)
    
    # Extract and save backbone for future Rad-FID computation
    backbone_path = os.path.join(args.output_dir, "backbone_only.pth")
    torch.save(model.vit.state_dict(), backbone_path)
    
    print(f"Success! Model and backbone saved to {args.output_dir}")

if __name__ == "__main__":
    main()
