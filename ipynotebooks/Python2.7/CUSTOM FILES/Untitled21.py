# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

def asd:    
    
    for i in array_order:
        try:
            if i == 'p_gpcp':            
                coverageID = 'None'
                #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):
        except Exception, e:
            continue
            
        try:                
            if i == 'p_trmm':
                coverageID = 'trmm_3b42_coverage_1'
                #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):                
        except Exception, e:
            continue            

        try:             
            if i == 't_lst':
                coverageID = 'modis_11c2_cov'
                request_name = i
                getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory)
        except Exception, e:
            continue            
            
        try:
            if i == 'et_radi':
                coverageID = 'radi_et_v1'
                #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):
        except Exception, e:
            continue              
            
        try:            
            if i == 'ndvi_modis':
                coverageID = 'modis_13c1_cov'
                request_name = i        
                getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory)
        except Exception, e:
            continue              

        try:
            if i == 'di_pap':
                coverageID = 'gpcp'
                request_name = i        
                PRECIP_DI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        except Exception, e:
            continue
            
        try:            
            if i == 'di_vci':
                coverageID = 'modis_13c1_cov'
                request_name = i        
                _VCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        except Exception, e:
            continue              

        try:
            if i == 'di_tci':
                coverageID = 'modis_11c2_cov'
                request_name = i        
                _TCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        except Exception, e:
            continue

        try:
            if i == 'di_vhi':
                coverageID = 'modis_11c2_cov'
                request_name = i        
                _VHI_CAL(coverageID, request_name, from_date_order,bbox_order, directory,alpha = 0.5)
        except Exception, e:
            continue              

        try:
            if i == 'di_nvai':
                coverageID = 'modis_11c2_cov'
                request_name = i        
                _NVAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)                
        except Exception, e:
            continue              

        try:
            if i == 'di_ntai':
                coverageID = 'modis_11c2_cov'
                request_name = i        
                _NTAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        except Exception, e:
            continue              

        try:            
            if i == 'di_netai':
                coverageID = 'modis_11c2_cov'
                #request_name = i        
                #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory): 
        except Exception, e:
            continue              

