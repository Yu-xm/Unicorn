import pickle
import torch
import os
import logging
import numpy as np

def setup_logging():

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("compute_text_mean.log"),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    
    TEXT_EMBED_MEAN = "/data/xmyu/data/embeddings/pkl/text_embed_mean_shift_512_47.pkl"
    input_files = [f"/data/xmyu/data/embeddings/pkl/captions_512_47/caption_embeddings_{i}.pkl" for i in range(1, 8)]
    output_dir = "/data/xmyu/data/embeddings/pkl/captions_512_47_mean_shift/"
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Output directory created: {output_dir}")
        except Exception as e:
            logging.error(f"Failed to create output directory {output_dir}: {e}")
            return
    
    text_mean = torch.zeros(1, 1280)
    
    total_captions = 0
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            logging.warning(f"File does not exist: {file_path}")
            continue
        
        print(f"Loading data from {file_path}")
        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load file {file_path}: {e}")
                continue
        
        for item in data:
            embed = item.get("embed")
            if embed is None:
                logging.warning(f"Missing 'embed' key in item: {item}")
                continue
            try:
                cap_embed = torch.from_numpy(embed).float()
            except Exception as e:
                logging.warning(f"Failed to convert 'embed' to tensor: {e}")
                continue
            text_mean += cap_embed.unsqueeze(0)
            total_captions += 1
    
    if total_captions == 0:
        logging.error("No valid text embeddings found for calculation.")
        return
    
    text_mean = text_mean / total_captions
    
    try:
        with open(TEXT_EMBED_MEAN, "wb") as f:
            pickle.dump(text_mean, f)
        print(f"Text embedding mean saved to {TEXT_EMBED_MEAN}")
        print(f"Processed a total of {total_captions} captions.")
    except Exception as e:
        logging.error(f"Failed to save mean to file {TEXT_EMBED_MEAN}: {e}")
        return
    
    text_mean_np = text_mean.numpy().squeeze(0)
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            logging.warning(f"File does not exist: {file_path}")
            continue
        
        print(f"Processing and saving data from {file_path}")
        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load file {file_path} for normalization: {e}")
                continue
        
        modified = False
        for item in data:
            embed = item.get("embed")
            if embed is None:
                logging.warning(f"Missing 'embed' key in item: {item}")
                continue
            try:
                embed = np.array(embed)
                if embed.shape != text_mean_np.shape:
                    logging.warning(f"'embed' shape {embed.shape} does not match mean shape {text_mean_np.shape}, skipping item.")
                    continue
                embed_mean_shift = embed - text_mean_np
                item['embed'] = embed_mean_shift
                modified = True
            except Exception as e:
                logging.warning(f"Failed to normalize 'embed': {e}")
                continue
        
        if modified:
            file_name = os.path.basename(file_path)
            new_file_path = os.path.join(output_dir, file_name)
            try:
                with open(new_file_path, "wb") as f:
                    pickle.dump(data, f)
                print(f"Normalized data saved to {new_file_path}")
            except Exception as e:
                logging.warning(f"Failed to save normalized data to file {new_file_path}: {e}")
        else:
            logging.warning(f"No 'embed' data to modify in file {file_path}.")

if __name__ == "__main__":
    main()
