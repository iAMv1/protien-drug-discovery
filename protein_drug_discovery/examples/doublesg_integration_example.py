"""
DoubleSG-DTA Integration Example
Compleases
    )ghts & Bise Weit to u wanif youet to True   # Sndb=False      use_wa  ce,
ice=deviev  d    =1e-4,
  _raterning    leaency
    ory efficir memze fo batch sialler  # Smsize=16,atch_  b      dataset,
al_l_dataset=v       va
 ,rain_datasetet=tn_datastrai     del,
   moodel=  mer(
      inDTATra= DoubleSGiner r
    traainetralize  5: Initi
    # Step)
    }"meters()):,.para in modell() for p{sum(p.nume: arametersodel print(f"M
    p
     )im=256
   idden_d     final_heads=8,
   ntion_h      attes=3,
  in_layer,
        gdden_dim=128    gin_hiion
     dimenshidden-2 8M =320,  # ESMidden_dim  esm_h
      im=78,g_feature_d  dru,
      delm_mo=es  esm_model
      el(SGDTAMod= Doubleel     modl...")
G-DTA modeleSing Doublizint("Initia
    pr-DTA modeled DoubleSGize enhanc: Initialep 4 # St)
    
   aset)}"_daten(testTest: {lset)}, len(val_datat)}, Val: {n_datase{len(trai Train: t sizes -asent(f"Datpri
       
    )
 erir, tokeniz     data_dasets(
   _datesgte_doublaset = creaatst_daset, te, val_datetatasin_d
    traasetsCreate dat  # Step 3:    
  
 es=10000)ax_sampl_pipeline(m_full.process processora_dir =
    # datessor()gDBProcin Bindrocessor =)
    # p usetoent mm (uncondingDB datas real Bion B: Proces Opti
    
    #1000)m_samples=a(nundingdb_date_mock_bir = creatata_disting
    d data for te: Use mockon A
    # Opti")
    aset...paring dat print("Predataset
   e or load  Creat Step 2:  
    #R50D")
  _t6_8M_Uk/esm2boo"facetrained(preenizer.from_er = EsmToktokenizodel()
    _manager.load_mel = esmsm_mod)
    eManager(r = ESMModel_manageesm    el...")
modng ESM-2 rint("Loadimodel
    pize ESM-2 Initial  # Step 1: 
  )
    device}"g device: {inint(f"Us    pr 'cpu')
sele() els_availabuda.itorch.ce('cuda' if evictorch.de =     devicion
onfigurat   # C""
    
 n"nctioxample fu""Main e
    "n():
def mair
MModelManagel import ESesm_modery.core._discoverugrotein_drom p
f_data
)indingdbock_bte_msor, creangDBProces  Bindirt (
  mposor idb_procesbindingcovery.data._drug_disrotein
from pdatasets
)sg_eate_double crt,taseAffinityDa DrugTargetrainer,DoubleSGDTAT    import (
 sg_trainerubleng.dovery.traini_drug_discoinom prote
)
frsorceshProolecularGrap, MSGDTAModel  Doublet (
  ion imporegratlesg_intdels.doub.modiscoveryrotein_drug_om p)))

fr__file__)th(spaath.abrname(os.path.dime(os.pth.dirnaappend(os.pays.path.h
sy to pattorirecnt dAdd pare

# 
import sys os
importy as npnump
import s as pdmport pandal
ideer, EsmMosmTokenizs import Eormer
from transfnnch.nn as 
import torort torch

imp2
"""ith ESM-model w-DTA SGDoubleanced use the enhwing how to ple sho examte