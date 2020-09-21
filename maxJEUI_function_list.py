def Sturges_quantization(data):
    import pandas as pd
    import numpy as np
    max_value=data.max().max()
    min_value=data.min().min()
    bin_width=(max_value-min_value)/(1+np.log2(data.shape[0]))
    Sturges_binedges=np.arange(min_value, max_value, bin_width)
    Sturges_binedges = np.append(Sturges_binedges, max_value)
    Sturges_center= [ (Sturges_binedges[i]+bin_width/2) for i in np.arange(len(Sturges_binedges))]
    bin_num=len(Sturges_binedges)-1
    inds = np.digitize(data, Sturges_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    columnns_name=data.columns
    Sturges_quantized= pd.DataFrame(inds,columns=columnns_name)
    Sturges_quantized_value=Sturges_quantized.replace(to_replace = (np.arange(len(Sturges_center))+1), value =Sturges_center) 
    return Sturges_quantized,Sturges_quantized_value, Sturges_center, Sturges_binedges 

def Scott_quantization(data):
    import pandas as pd
    import numpy as np
    max_value=data.max().max()
    min_value=data.min().min()
    all_1D=data.values.flatten()
    bin_width=(3.49*all_1D.std())/(data.shape[0])**(1/3)
    Scott_binedges=np.arange(min_value, max_value, bin_width)
    Scott_binedges = np.append(Scott_binedges, max_value)
    bin_num=len(Scott_binedges)-1
    Scott_center= [ (Scott_binedges[i]+bin_width/2) for i in np.arange(len(Scott_binedges))]
    inds = np.digitize(data, Scott_binedges)
    # np.digitize discritize x for bins[i-1] <= x < bins[i]
    inds = np.where(inds > bin_num,bin_num,inds) 
    # ==> if x==max_value, it put x on new bins(bin_num+1) ==>np.where is used to put x==max_value in the last bin(bin_num)
    columnns_name=data.columns
    Scott_quantized= pd.DataFrame(inds,columns=columnns_name)
    Scott_quantized_value=Scott_quantized.replace(to_replace = (np.arange(len(Scott_center))+1), value =Scott_center) 
    return Scott_quantized,Scott_quantized_value, Scott_center, Scott_binedges 

def floor_quantization(data,a_f):
    import pandas as pd
    import numpy as np
    floor_global_quantized= 1+((2*data+a_f)/(2*a_f)).apply(np.floor)
    max_value=data.max().max()
    min_value=data.min().min()
    binedges=[min_value, a_f/2]
    i=0
    max_binedges_floor=max(binedges)
    while max_binedges_floor <max_value:
      binedges.append(binedges[i+1]+a_f)
      max_binedges_floor=max(binedges)
      i+=1
    global_centers =[]
    for i in np.arange(len(binedges)):
            if i<len(binedges)-1:
                global_centers.append(np.squeeze((binedges[i]+binedges[i+1])/2).tolist())
    floor_global_quantized_value=floor_global_quantized.replace(to_replace = (np.arange(len(global_centers))+1), value =global_centers)  
    return floor_global_quantized,floor_global_quantized_value, global_centers, binedges


def entropy_1D(signal_qauntized):
    import numpy as np
    (unique, counts) = np.unique(signal_qauntized, return_counts=True)
    prob_dist=counts/sum(counts)
    H_x=[]
    for i in np.arange(len(prob_dist)):
            H_x.append(np.squeeze(-prob_dist[i]*(np.log2(prob_dist[i]))).tolist())
    Entropy=np.nansum(H_x)
    return Entropy

def merging_operator_2varibles(signal1,signal2):
    import pandas as pd
    import numpy as np
    signal1=list(map(str, signal1))
    signal2=list(map(str, signal2))
    merg=list(map(''.join, zip(signal1, signal2)))
    merg=list(map(lambda num: int(num), merg))
    unique_elements=np.unique(merg)
    merg_signal = pd.DataFrame(merg)
    # relabeling
    merg_signal.replace(to_replace=unique_elements, value=(np.arange(len(unique_elements))+1))
    return merg_signal

def info_stats_merging_operator_multiple_varibles(data,order):
    import pandas as pd
    import numpy as np
    previous_merg=[]
    H_marg=[]
    previous_merg_2nd=[]
    for i in order:
        H_marg.append(entropy_1D(data.iloc[:,i-1]))
    for i in order:
        if i==order[0]:
            previous_merg=merging_operator_2varibles(data.iloc[:,i-1].astype('int64'),data.iloc[:,i-1].astype('int64')).squeeze()
        else:
            previous_merg_2nd=previous_merg
            previous_merg=merging_operator_2varibles(previous_merg,data.iloc[:,i-1].astype('int64')).squeeze()

    Joint_entrpoy=entropy_1D(previous_merg)
    total_correlation_last_added=sum(H_marg)-entropy_1D(previous_merg)
    mutual_last_added=H_marg[-1]-(entropy_1D(previous_merg)-entropy_1D(previous_merg_2nd))
    return Joint_entrpoy,mutual_last_added,total_correlation_last_added

def greedy_optimizer_maxJE(data):
    import numpy as np
    import itertools
    idx_left=list(np.arange(data.shape[1])+1)
    idx_selected=[]
    #  Finding First Station
    H_marg=[]
    for i in idx_left:
        H_marg.append(entropy_1D(data.iloc[:,i-1]))
    index_sorted=sorted(range(len(H_marg)), key=lambda k: H_marg[k],reverse=True)
    idx_new_select=idx_left[index_sorted[0]]
    idx_selected.append(idx_new_select)
    idx_left.remove(idx_new_select)
    #  Finding ranking for 2nd to end
    for j in np.arange(data.shape[1]-1):
        search=[]
        for i in idx_left :
            combination_order=[idx_selected,[i]]
            combination_order= list(itertools.chain.from_iterable(combination_order))
            search.append(info_stats_merging_operator_multiple_varibles(data,combination_order)[0])
        index_sorted=sorted(range(len(search)), key=lambda k: search[k],reverse=True)
        idx_new_select=idx_left[index_sorted[0]]
        idx_selected.append(idx_new_select)
        idx_left.remove(idx_new_select)
    Ranked_order=idx_selected
    return Ranked_order

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def information_decomposition (source1,source2,target):
    import numpy as np
    np.seterr(all="ignore")
    source1=source1.astype(int)
    source2=source2.astype(int)
    target=target.astype(int)
    bin_used=max(source1.max(),source2.max(),target.max())
    mesh_s1_s2=np.zeros((bin_used, bin_used))
    mesh_s1_T=np.zeros((bin_used, bin_used))
    mesh_s2_T=np.zeros((bin_used, bin_used))
    mesh_s1_s2_T=np.zeros((bin_used, bin_used, bin_used))
    for i in np.arange(len(source1)):
        ii=source1[i]-1
        jj=source2[i]-1
        kk=target[i]-1
        mesh_s1_s2[ii,jj]=mesh_s1_s2[ii,jj]+1
        mesh_s1_T[ii,kk]=mesh_s1_T[ii,kk]+1
        mesh_s2_T[jj,kk]=mesh_s2_T[jj,kk]+1
        mesh_s1_s2_T[ii,jj,kk]=mesh_s1_s2_T[ii,jj,kk]+1
    prob_s1_s2=mesh_s1_s2/sum(sum(mesh_s1_s2))   
    prob_s1_T=mesh_s1_T/sum(sum(mesh_s1_T))
    prob_s2_T=mesh_s2_T/sum(sum(mesh_s2_T))
    prob_s1_s2_T=mesh_s1_s2_T/sum(sum(sum(mesh_s1_s2_T)))
    marg_prob_s1=prob_s1_s2_T.sum(axis=2).sum(axis=1)
    marg_prob_s2=prob_s1_s2_T.sum(axis=2).sum(axis=0)
    marg_prob_T=prob_s1_s2_T.sum(axis=1).sum(axis=0)
    MI_s1_s2_T=np.zeros((bin_used, bin_used, bin_used))
    for i in np.arange(bin_used):
        for j in np.arange(bin_used):
            for k in np.arange(bin_used):
                    MI_s1_s2_T[i,j,k]=prob_s1_s2_T[i,j,k]*np.log2(prob_s1_s2_T[i,j,k]/(prob_s1_s2[i,j]*marg_prob_T[k])) 
    # MI_s1_s2_T stands for I(s1,s2;tar) refrenced to GK2016 paper doi:10.1002/2016WR020216.
    MI_s1_s2_T=np.nansum(np.nansum(np.nansum(MI_s1_s2_T)))
    MI_s1_T=np.zeros((bin_used, bin_used))
    for i in np.arange(bin_used):
        for j in np.arange(bin_used):
                    MI_s1_T[i,j]=prob_s1_T[i,j]*np.log2(prob_s1_T[i,j]/(marg_prob_s1[i]*marg_prob_T[j])) 
    MI_s1_T=np.nansum(np.nansum(MI_s1_T))
    MI_s2_T=np.zeros((bin_used, bin_used))
    for i in np.arange(bin_used):
        for j in np.arange(bin_used):
                    MI_s2_T[i,j]=prob_s2_T[i,j]*np.log2(prob_s2_T[i,j]/(marg_prob_s2[i]*marg_prob_T[j])) 
    MI_s2_T=np.nansum(np.nansum(MI_s2_T))
    # MI_s1_T_given_s2 stands for I(s1,tar|s2)=I(s1,s2;tar)-I(s2,T)
    MI_s1_T_given_s2=MI_s1_s2_T-MI_s2_T
    # II stands for interaction information refrenced to GK2016 paper doi:10.1002/2016WR020216.
    II=MI_s1_T_given_s2-MI_s1_T
    if -II>0:
        R_min=-II
    else:
        R_min=0
    R_MMI=min(MI_s1_T,MI_s2_T)
    H_s1=np.zeros(len(marg_prob_s1))
    for i in np.arange(len(marg_prob_s1)):
        H_s1[i]=-marg_prob_s1[i]*np.log2(marg_prob_s1[i])
    H_s1=np.nansum(H_s1)
    H_s2=np.zeros(len(marg_prob_s2))
    for i in np.arange(len(marg_prob_s1)):
        H_s2[i]=-marg_prob_s2[i]*np.log2(marg_prob_s2[i])
    H_s2=np.nansum(H_s2)
    MI_s1_s2=np.zeros((bin_used, bin_used))
    for i in np.arange(bin_used):
        for j in np.arange(bin_used):
                    MI_s1_s2[i,j]=prob_s1_s2[i,j]*np.log2(prob_s1_s2[i,j]/(marg_prob_s1[i]*marg_prob_s2[j])) 
    MI_s1_s2=np.nansum(np.nansum(MI_s1_s2))
    I_s=MI_s1_s2/min(H_s1,H_s2)
    R_s=R_min+I_s*(R_MMI-R_min)
    Unique_s1=MI_s1_T-R_s
    synergy=II-R_s
    return Unique_s1,synergy,R_s,II,MI_s1_T

def greedy_optimizer_maxJEUI(data):
    import numpy as np
    import itertools
    import copy
    idx_left=list(np.arange(data.shape[1])+1)
    idx_selected=[]
    #  Finding First Station
    H_marg=[]
    for i in idx_left:
            H_marg.append(entropy_1D(data.iloc[:,i-1]))
    index_sorted_H_marg=sorted(range(len(H_marg)), key=lambda k: H_marg[k],reverse=True)
    idx_new_select=idx_left[index_sorted_H_marg[0]]
    H_last_iteration=H_marg[index_sorted_H_marg[0]]
    idx_selected.append(idx_new_select)
    idx_left.remove(idx_new_select)
    #  Finding ranking for 2nd to end
    for j in np.arange(data.shape[1]-1):
        search=[]
        for i in idx_left:
            combination_order=[idx_selected,[i]]
            combination_order= list(itertools.chain.from_iterable(combination_order))
            search.append(info_stats_merging_operator_multiple_varibles(data,combination_order)[0])     
        index_sorted_JE=sorted(range(len(search)), key=lambda k: search[k],reverse=True)

    ##### condition1 ##############
        if len(search)>1 and search[index_sorted_JE[0]]==search[index_sorted_JE[1]]:
            combination_order1=[idx_selected,[idx_left[index_sorted_JE[0]]]]
            combination_order1= list(itertools.chain.from_iterable(combination_order1))
            mutual1=info_stats_merging_operator_multiple_varibles(data,combination_order1)[1]
            combination_order2=[idx_selected,[idx_left[index_sorted_JE[1]]]]
            combination_order2= list(itertools.chain.from_iterable(combination_order2))
            mutual2=info_stats_merging_operator_multiple_varibles(data,combination_order2)[1]
            if mutual1==mutual2:
                unique_info_idx_left=np.zeros((len(idx_left), len(idx_selected), len(idx_selected)-1))
                i=0
                j=0
                k=0
                for s1 in idx_left:
                    target_index=copy.deepcopy(idx_selected)
                    j=0
                    for T in target_index:
                        source_2nd=copy.deepcopy(idx_selected)
                        source_2nd.remove(T)
                        k=0
                        for s2 in source_2nd:
                            unique_info_idx_left[i,j,k]=information_decomposition(data.iloc[:,s1-1],data.iloc[:,s2-1],data.iloc[:,T-1])[0]
                            k+=1
                        j+=1
                    i+=1
                AUI_idx_left=unique_info_idx_left.mean(axis=2).mean(axis=1)
                index_sorted_AUI=sorted(range(len(AUI_idx_left)), key=lambda k: AUI_idx_left[k],reverse=True)
                idx_new_select=idx_left[index_sorted_AUI[0]]
                idx_selected.append(idx_new_select)
                H_last_iteration=info_stats_merging_operator_multiple_varibles(data,idx_selected)[0]
                idx_left.remove(idx_new_select)
    ##### condition2 ##############
        if ((search[index_sorted_JE[0]]-H_last_iteration)/H_last_iteration)<.01:
            unique_info_idx_left=np.zeros((len(idx_left), len(idx_selected), len(idx_selected)-1))
            i=0
            j=0
            k=0
            for s1 in idx_left:
                target_index=copy.deepcopy(idx_selected)
                j=0
                for T in target_index:
                    source_2nd=copy.deepcopy(idx_selected)
                    source_2nd.remove(T)
                    k=0
                    for s2 in source_2nd:
                        unique_info_idx_left[i,j,k]=information_decomposition(data.iloc[:,s1-1],data.iloc[:,s2-1],data.iloc[:,T-1])[0]
                        k+=1
                    j+=1
                i+=1
            AUI_idx_left=unique_info_idx_left.mean(axis=2).mean(axis=1)
            index_sorted_AUI=sorted(range(len(AUI_idx_left)), key=lambda k: AUI_idx_left[k],reverse=True)
            idx_new_select=idx_left[index_sorted_AUI[0]]
            idx_selected.append(idx_new_select)
            H_last_iteration=info_stats_merging_operator_multiple_varibles(data,idx_selected)[0]
            idx_left.remove(idx_new_select)
        else:
            idx_new_select=idx_left[index_sorted_JE[0]]
            idx_selected.append(idx_new_select)
            H_last_iteration=info_stats_merging_operator_multiple_varibles(data,idx_selected)[0]
            idx_left.remove(idx_new_select)
    Ranked_order=idx_selected
    return Ranked_order