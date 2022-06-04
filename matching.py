def matching (x, P, Z, R, M, g):
    n_measurs = Z.shape(1)
    n_map = M.shape(1)

    d = np.zeros((n_measurs, n_map))
    v = np.zeros((2, n_measurs * n_map))
    H = np.zeros((2, 3, n_measurs * n_map ))

    for aux_nme in range(0, n_measurs):
        for aux_nmap in range(0, n_map):
            z_predict, H




    v = v
    H =  H
    R_seg = R_seg(:, :, measursidx)
    return v, H, R_seg