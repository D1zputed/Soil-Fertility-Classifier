"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import csv
import traceback

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
def write_result_csv(csv_file_name, 
                     bat_s, 
                     lr, 
                     epoch, 
                     dropout, 
                     pretrained, 
                     train_l, 
                     train_acc, 
                     test_l, 
                     test_acc):
  """
  This function simply saves the results to the csv
  """
  try:
    with open(csv_file_name, mode='a', newline='') as file:    
      writer = csv.writer(file, delimiter='\t')

      train_l = f"{train_l[-1]:.4f}"
      train_acc = f"{train_acc[-1]:.4f}"
      test_l = f"{test_l[-1]:.4f}"
      test_acc = f"{test_acc[-1]:.4f}"
      
      writer.writerow([bat_s, lr, epoch, dropout, pretrained, train_l, train_acc, test_l, test_acc])  
      print(f"Inserted new data in {csv_file_name}")
  except Exception as e:
    print(e)
    traceback.print_exc()



