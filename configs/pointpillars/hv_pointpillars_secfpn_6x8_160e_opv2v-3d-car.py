_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_opv2v.py',
    '../_base_/datasets/opv2v-3d-car.py',
    '../_base_/schedules/cyclic_80e.py', 
    '../_base_/default_runtime.py'
]