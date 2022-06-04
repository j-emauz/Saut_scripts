def matching (x, P, Z, R, M, g):
    #Z: observations measurements
    n_measurs = Z.shape(1)
    n_map = M.shape(1)

    d = np.zeros((n_measurs, n_map))
    v = np.zeros((2, n_measurs * n_map))
    H = np.zeros((2, 3, n_measurs * n_map ))

    for aux_nme in range(0, n_measurs):
        for aux_nmap in range(0, n_map):
            Z_predict, H[:, :, aux_nmap + (aux_nme -1) * n_map] = measurementfunction(x, M[:, n_map])
            v[:, aux_nmap + (aux_nme -1) * n_map] = Z[:, aux_nme] - Z_predict
            W = H[:, :, aux_nmap + (aux_nme -1) * n_map] * P * np.transpose(H[:, :, aux_nmap + (aux_nme -1) * n_map]) + R_seg[:, :, aux_nme]
            d[aux_nme, aux_nmap] = np.transpose(v[:, aux_nmap + (aux_nme -1) * n_map]) * np.linalg.inv(W) * v[:, aux_nmap + (aux_nme -1) * n_map]

    minima, mapidx = (np.transpose(d)).min(0),(np.transpose(d)).argmin(0)
    measursidx = np.argwhere(minima < g**2)
    mapidx = mapidx(measursidx)

    v = v[:, mapidx + (measursidx -1)* n_map]
    H =  H[:, :,  mapidx + (measursidx -1)* n_map]
    R_seg = R_seg[:, :, measursidx]
    return v, H, R_seg