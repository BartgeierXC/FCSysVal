# =============================================================================
# FCS Measurement Variable Name Constants
# Central definition of all column name strings used across notebooks.
# Usage:  from data.parameters.fcs_variable_naming import *
# =============================================================================

# --- Electrical ---
I_A             = 'I.S [A]'
I_A_set         = 'I.S.set [A]'
I_A_act         = 'I.S.act [A]'
I_min_gf_A      = 'I.S.MinGasFlow.set [A]'
U_V             = 'U.S [V]'
U_cell_V        = 'U.S.AveCell [V]'
U_cell_max_V    = 'U.S.MaxCell [V]'
U_cell_min_V    = 'U.S.MinCell [V]'
j_A_per_cm2     = 'j.S [A/cm²]'
j_A_per_cm2_set = 'j.S.set [A/cm²]'
P_el_kW         = 'P.S.el [kW]'
P_th_W      = 'P.S.th [W]'

# --- Stoichiometry / Concentration ---
stoic_An        = 'Stoic.S.A'
stoic_An_set    = 'Stoic.S.A.set'
stoic_Ca        = 'Stoic.S.C'
stoic_Ca_set    = 'Stoic.S.C.set'
stoic_air       = 'Stoic.S.C'       # alias
lambda_K        = 'Stoic.S.C'       # alias used in validation notebooks
conc_H2_pct         = 'Conc.Si.H.A [%]'
conc_H2_pct_set     = 'Conc.Si.H.A.set [%]'
conc_N2_An_pct      = 'Conc.Si.N.A [%]'
conc_N2_An_pct_set  = 'Conc.Si.N.A.set [%]'
conc_air_pct        = 'Conc.Si.Air.C [%]'
conc_air_pct_set    = 'Conc.Si.Air.C.set [%]'
conc_N2_Ca_pct      = 'Conc.Si.N.C [%]'
conc_N2_Ca_pct_set  = 'Conc.Si.N.C.set [%]'

# --- Anode flow rates ---
vf_An_nl_per_min        = 'F.Si.Total.A [nl/min]'
vf_An_nl_per_min_set    = 'F.Si.Total.A.set [nl/min]'
vf_h2_nl_per_min        = 'F.Si.H.A [nl/min]'
vf_h2_nl_per_min_set    = 'F.Si.H.A.set [nl/min]'
vf_n2_An_nl_per_min     = 'F.Si.N.A [nl/min]'
vf_n2_An_nl_per_min_set = 'F.Si.N.A.set [nl/min]'
mf_An_kg_per_s          = 'F.Si.A.Total [kg/s]'

# --- Cathode flow rates ---
vf_Ca_nl_per_min        = 'F.Si.Total.C [nl/min]'
vf_Ca_nl_per_min_set    = 'F.Si.Total.C.set [nl/min]'
Vf_air_nl_per_min       = 'F.Si.Total.C [nl/min]'  # alias used in cal notebooks
vf_air_hi_nl_per_min    = 'F.Si.hi.Air.C [nl/min]'
vf_air_hi_nl_per_min_set = 'F.Si.hi.Air.C.set [nl/min]'
vf_air_lo_nl_per_min    = 'F.Si.lo.Air.C [nl/min]'
vf_air_Ca_nl_per_min    = 'F.Si.Air.C [nl/min]'
vf_air_lo_nl_per_min_set = 'F.Si.lo.Air.C.set [nl/min]'
vf_n2_Ca_nl_per_min     = 'F.Si.N.C [nl/min]'
vf_n2_Ca_nl_per_min_set = 'F.Si.N.C.set [nl/min]'
mf_Ca_kg_per_s          = 'F.Si.C.Total [kg/s]'
mf_Cmp_kg_per_h     = 'F.Si.Cmp.C [kg/h]'
mf_kg_per_h             = 'F.Si.C [kg/h]'

# --- Coolant flow rates ---
vf_CL_nl_per_min    = 'F.Si.CL [nl/min]'
vf_CL_nl_per_min_set = 'F.Si.CL.set [nl/min]'
vf_w_nl_per_min     = 'F.Si.CL [nl/min]'   # alias used in 12h
s_CL_uS_per_cm      = 's.Si.CL [µS/cm]'

# --- Humidity ---
phi_An_pct          = 'RH.Si.A.CV [%]'
phi_An_calc_pct     = 'anode_inlet_rel_hum [%]'
phi_Ca_pct          = 'cathode_inlet_rel_hum [%]'
RH_pct              = 'cathode_inlet_rel_hum [%]'  # alias used in 13_validate
phi_Ca_cv_pct       = 'RH.Si.C.CV [%]'
rh_air_in_pct       = 'RH.Si.C.CV [%]'     # alias used in 12b
rh_Si_Ca_pct            ='RH.Si.C [%]'
phi_pct             = 'RH.Si.A.CV [%]'     # alias used in 12g
DPT_Si_An_degC      = 'DPT.Si.A [°C]'
DPT_Si_An_degC_set  = 'DPT.Si.A.set [°C]'
DPT_Si_An_calc_degC = 'temp_anode_dewpoint [°C]'
DPT_Si_Ca_degC      = 'DPT.Si.C [°C]'
DPT_Si_Ca_degC_set  = 'DPT.Si.C.set [°C]'
DPT_Si_Ca_calc_degC = 'temp_cathode_dewpoint [°C]'
rh_an_01_pct        = 'rh_calc.anode_01 [%]'
rh_an_02_pct        = 'rh_calc.anode_02 [%]'
rh_ca_01_pct        = 'rh_calc.cathode_01 [%]'
rh_ca_02_pct        = 'rh_calc.cathode_02 [%]'

# --- Anode temperatures ---
T_Si_An_degC        = 'T.Si.A [°C]'
T_Si_An_degC_set    = 'T.Si.A.set [°C]'
T_So_An_degC        = 'T.So.A [°C]'

# --- Cathode temperatures ---
T_Si_Ca_degC        = 'T.Si.C [°C]'
T_Si_C_degC         = 'T.Si.C [°C]'        # alias used in cal notebooks
T_Si_Ca_degC_set    = 'T.Si.C.set [°C]'
T_So_Ca_degC        = 'T.So.C [°C]'
T_Hu_i_degC     = 'T.Hu.i [°C]'

# --- Coolant temperatures ---
T_Si_CL_degC        = 'T.Si.CL [°C]'
T_Si_Cl_degC        = 'T.Si.CL [°C]'       # alias used in cal/val notebooks
T_Si_CL_degC_set    = 'T.S.CL.set [°C]'
T_So_CL_degC        = 'T.So.CL [°C]'
T_So_Cl_degC        = 'T.So.CL [°C]'       # alias used in cal/val notebooks
DT_CL_K             = 'DT.S.CL [K]'
DT_CL_K_set         = 'DT.S.CL.set [K]'
DT2_CL_K            = 'DT2.S.CL [K]'

# --- Anode pressures ---
p_Si_An_barg    = 'p.Si.A [barg]'
p_So_An_barg    = 'p.So.A [barg]'
p_Si_A_bara     = 'p.Si.A [bara]'
p_So_A_bara     = 'p.So.A [bara]'
p_Si_A_set_bara = 'p.Si.A.set [bara]'
p_An_barg_set   = 'p.S.A.set [barg]'
pD_An_mbar      = 'pD.S.A [mbar]'
dp_an_mbar      = 'pD.S.A [mbar]'      # alias used in cal notebooks
p_an_in_barg    = 'p.Si.A [barg]'      # alias used in cal notebooks
p_h2_in_barg    = 'p.Si.A [barg]'      # alias used in validation notebooks

# --- Cathode pressures ---
p_Si_Ca_barg    = 'p.Si.C [barg]'
p_So_Ca_barg    = 'p.So.C [barg]'
p_Si_Ca_bara    = 'p.Si.C [bara]'
p_So_Ca_bara    = 'p.So.C [bara]'
p_Si_Ca_set_bara = 'p.Si.C.set [bara]'
p_Ca_barg_set   = 'p.S.C.set [barg]'
pD_Ca_mbar      = 'pD.S.C [mbar]'
p_air_in_barg   = 'p.Si.C [barg]'      # alias used in cal notebooks
p_air_out_barg  = 'p.So.C [barg]'      # alias used in cal notebooks

# --- Coolant pressures ---
p_Si_CL_barg    = 'p.Si.CL [barg]'
p_So_CL_barg    = 'p.So.CL [barg]'
p_CL_barg_set   = 'p.S.CL.set [barg]'
pD_CL_mbar      = 'pD.S.CL.CV [mbar]'

# --- Auxiliary pressures ---
p_Hu_An_barg    = 'p.Hu.o.A [barg]'
p_Hu_Ca_barg    = 'p.Hu.o.C [barg]'
p_N2_lab_barg   = 'p.N.Lab [barg]'
p_DW_lab_barg   = 'p.DW.Lab [barg]'
p_fuel          = 'pressure_fuel_supply'

# --- Valve states ---
x_BV_So_C_AO_pct = 'BV.So.C.AO [%]'
x_Fr_Mix_CL_AO_pct = 'Fr.Mix.CL.AO [%]'
x_Fr_Mix_CL_AI_pct = 'Fr.Mix.CL.AI [%]'
x_Fr_Hu_i_AO_pct = 'Fr.Hu.i.AO [%]'

# --- ARB (Air Recirculation Blower) ---
v_ARB_rpm       = 'v.ARB.act [rpm]'
v_ARB_req_rpm   = 'v.ARB.req [rpm]'
v_ARB_set_rpm = 'Fr_Pu_AR_set [%]'
P_ARB_kW        = 'P.ARB.CV [kW]'

# --- CMP ---
v_CMP_rpm = 'v.CMP.act [rpm]'
v_CMP_set_rpm = 'Fr_CMP_C_set [rpm]'
p_CMP_o_bara = 'p.CMP.o.C [bara]'
p_CMP_i_bara = 'p.CMP.i.C [bara]'
F_Si_Air_C_set_l_min    = 'F_Si_Air_C_set [l/min]'
SS_vf_air_set_nl_per_min = 'SS.F.Si.Air.C.set [nl/min]'
P_CMP_kW = 'P.CMP.CV [kW]'
T_CMP_o_degC = 'T.CMP.o [°C]'

#--- CWP ---
P_CWP_W = 'P.FU [W]'

# --- FCS specific variables ---
I_S_set_A           = 'I_S_set [A]'
p_S_A_set_bar       = 'p_S_A_set [bar]'
C_Stoic             = 'C_Stoic []'
C_Stoic_act         = 'Calc.C.Stoic.act []'
p_S_C_set_bar       = 'p_S_C_set [bar]'
dp_Pu_AR_set_mbar   = 'dp_Pu_AR_set [mbar]'
Delta_T_degC        = 'Delta_T [°C]'
T_Stack_degC        = 'T_Stack [°C]'
SS_T_Stack_degC     = 'SS.T.Stack [°C]'
SS_Delta_T_degC     = 'SS.Delta.T [°C]'
DPT_Si_C_pct        = 'DPT_Si_C [%]'
DPT_Si_C_degC        = 'DPT_Si_C [°C]'
CP_speed_pct            = 'CP_Speed [%]'
t_Pu_s                  = 'Purge_Duration [s]'
F_HE_Cl_l_min = 'F.HF.HE.CL [l/min]'
T_HF_o_CL_degC = 'T.HF.o.CL [°C]'
P_sys_kW = 'P.sys [kW]'
eta_sys_pct = 'eta_sys [%]'

# --- Metadata / state ---
profile_ID  = 'Profile_ID'
profile_name = 'Profile'
profile_step_no = 'Profile_StepNo [-]'
exp_id      = 'Experiment_ID'
state       = 'system.state'
state_set   = 'system.state_set'
sig_load_on = 'signal_out_load_on'
cal_hrs_h   = 'CalHrs [h]'
op_hrs_h    = 'OpHrs [h]'
