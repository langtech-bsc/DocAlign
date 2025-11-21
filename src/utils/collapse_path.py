def collapse_path_to_ranges(path):
    """
    Collapse path to handle many-to-many mappings.
    Groups consecutive items by whichever index changes.
    
    Returns:
        List of (source_group, target_group) where each is a list of indices
    """
    if not path:
        return []
    
    collapsed = []
    current_sources = [path[0][0]]
    current_targets = [path[0][1]]
    
    for i, j in path[1:]:
        # Check if either source or target changed
        source_changed = i != current_sources[-1]
        target_changed = j != current_targets[-1]
        
        if source_changed and target_changed:
            # Both changed: save current group, start new one
            collapsed.append((current_sources, current_targets))
            current_sources = [i]
            current_targets = [j]
        elif source_changed:
            # Only source changed: add to source group
            current_sources.append(i)
        elif target_changed:
            # Only target changed: add to target group
            current_targets.append(j)
        # else: both same, continue (shouldn't happen in valid DTW path)
    
    collapsed.append((current_sources, current_targets))
    return collapsed