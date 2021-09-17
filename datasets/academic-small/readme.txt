academic_full/
    Content:
        0: network in 1980
        ...
        35: network in 2015

    Format: 
        The adjacency list format described in https://github.com/luckiezhou/DynamicTriad

labels.pickle
    Content:
        A pickle file containing two lists.
        The first one is a list of numpy arrays, where the i-th (0-based) array is the label vector for nodes in year (i+1980)
        The second list contains a single numpy array, which is the overall label vector of all nodes.
    
    Order of the labels:
        Sort the nodes in ascending order w.r.t. their names (i.e. ID, as integer type), we'll obtain a list of nodes N.
        For each label vector L, the i-th label L_i corresponds to the i-th node N_i at the corresponding time step.
        For example, if there are three nodes {1, 11, 2} in total. 
        Given a label vector [-1, 3, 4], you'll concluded according to it 
            that users 1, 2, 11 are from communities 'Unknown', 'Theory', 'Data Mining' respectively.

    Value of the labels: 
        {-1: 'Unknown', 0: 'Architecture', 1: 'Computer Network', 2: 'Computer Security', 3: 'Data Mining', 4: 'Theory', 5: 'Graphics'}
