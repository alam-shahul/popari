import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import h5py

def subsample_h5_file(input_path, output_path, target_obs):
    adata = ad.read_h5ad(input_path)

    cell_types = adata.obs['cell_type'].unique()
    batches = adata.obs['batch'].unique()
    
    cells_per_type = target_obs // len(cell_types)
    remainder = target_obs % len(cell_types)

    indices_to_keep = []
    

    for i, cell_type in enumerate(cell_types):
        mask = adata.obs['cell_type'] == cell_type
        cells_of_type = np.where(mask)[0]
        
        
        cells_to_keep = cells_per_type + (1 if i < remainder else 0)
        
       
        if len(batches) > 1 and len(cells_of_type) >= len(batches):
            
            batch_indices = []
            for batch in batches:
                batch_mask = adata.obs['batch'] == batch
                combined_mask = mask & batch_mask
                batch_cells = np.where(combined_mask)[0]

                if len(batch_cells) > 0:
                    # Take cells per batch, roughly evenly distributed
                    cells_per_batch = max(1, cells_to_keep // len(batches))
                    # Don't take more cells than available in this batch
                    cells_per_batch = min(cells_per_batch, len(batch_cells))
                    # Randomly sample
                    selected = np.random.choice(batch_cells, size=cells_per_batch, replace=False)
                    batch_indices.extend(selected)
            
            # If we didn't get enough cells, take more randomly
            if len(batch_indices) < cells_to_keep:
                remaining_cells = list(set(cells_of_type) - set(batch_indices))
                if remaining_cells:
                    additional = min(cells_to_keep - len(batch_indices), len(remaining_cells))
                    batch_indices.extend(np.random.choice(remaining_cells, size=additional, replace=False))
            
            # If we got too many, subsample
            if len(batch_indices) > cells_to_keep:
                batch_indices = np.random.choice(batch_indices, size=cells_to_keep, replace=False)
            
            indices_to_keep.extend(batch_indices)
        else:
            # Simple case: just randomly sample from this cell type
            samples = min(cells_to_keep, len(cells_of_type))
            selected = np.random.choice(cells_of_type, size=samples, replace=False)
            indices_to_keep.extend(selected)
    
    # Create subsampled AnnData
    adata_sub = adata[indices_to_keep].copy()

    adata_sub.write_h5ad(output_path)
    return adata_sub

def print_h5_contents(filename):
   
    with h5py.File(filename, 'r') as f:
        print(f"File: {filename}")
        def visit_item(name, obj):
            indent = ' ' * (name.count('/') * 2)
            
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                # Print group attributes
                if len(obj.attrs) > 0:
                    print(f"{indent}  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"{indent}    {key}: {value}")
            
            elif isinstance(obj, h5py.Dataset):
                shape_str = str(obj.shape)
                dtype_str = str(obj.dtype)
                print(f"{indent}Dataset: {name}, Shape: {shape_str}, Type: {dtype_str}")
                
                if len(obj.attrs) > 0:
                    print(f"{indent}  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"{indent}    {key}: {value}")
                
                
                if obj.size < 10:
                    print(f"{indent}  Data: {obj[...]}")
                else:
                    try:
                        print(f"{indent}  First few elements: {obj[0:min(3, obj.shape[0])]}")
                    except:
                        pass  # Skip if we can't show a preview
        
        
        f.visititems(visit_item)
    return


if __name__ == "__main__":
    existing_sim_data = "tests/test_data/raehash_dataset/old_all_data.h5"
    print_h5_contents(existing_sim_data)
    new_data = "tests/test_data/raehash_dataset/all_data.h5"
    subsample_h5_file(existing_sim_data, new_data, 100)
    print_h5_contents(new_data)