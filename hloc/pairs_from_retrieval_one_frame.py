import torch
import numpy as np
import h5py

def pairs_from_retrieval_one_frame(
    query_desc, 
    db_names,
    db_descs_tensor,
    num_matched, 
    device='cpu'
):
    """
    Match a single query descriptor against a database of descriptors.
    
    Args:
        query_desc: numpy array or torch tensor of shape (D,) containing the query descriptor
        db_names: list of database image names
        db_descs_tensor: torch tensor of shape (N, D) containing database descriptors
        num_matched: number of top matches to return
        device: 'cpu' or 'cuda'
        
    Returns:
        List of tuples (db_name, score) sorted by score descending
    """
    
    # Process query descriptor
    if isinstance(query_desc, np.ndarray):
        query_desc = torch.from_numpy(query_desc).float()
    
    query_desc = query_desc.to(device)
    
    # Ensure query_desc is 2D (1, D)
    if query_desc.dim() == 1:
        query_desc = query_desc.unsqueeze(0)

    db_descs_tensor = db_descs_tensor.to(device)
        
    # Compute similarity
    # query_desc: (1, D), db_descs: (N, D) -> sim: (1, N)
    sim = torch.mm(query_desc, db_descs_tensor.t())
    
    # Get top k matches
    # Ensure we don't request more than available
    k = min(num_matched, sim.shape[1])
    scores, indices = torch.topk(sim, k, dim=1)
    
    scores = scores.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()
    
    matches = []
    for score, idx in zip(scores, indices):
        matches.append((db_names[idx], score))
        
    return matches
