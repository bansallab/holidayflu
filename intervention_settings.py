# January 25, 2017
# template for setting intervention parameters

# switch interventions on or off
experiment = 'yes' 

# options: 'yes', 'no'
# yes: there will be some interventions
# no: there will be no interventions

# indicate which school closure intervention to implement
disease_intervention = 'red_C_all' 

# options: 'none', 'red_C_all', 'red_C_cc', 'red_C_aa', 'red_C_ageROnly', 'red_C_ageROnly_less', 'red_C_ageROnly_more'
# none: no contact-related intervention 
# red_C_all: school closure intervention
# red_C_cc: reduce contact between children only (sensitivity)
# red_C_aa: reduce contact between adults only (sensitivity)
# red_C_ageROnly: partial school closure intervention (sensitivity)
# red_C_ageROnly_less: -10% contact rate from partial school closure (sensitivity)
# red_C_ageROnly_more: +10% contact rate from partial school closure (sensitivity)

# indicate which travel intervention to implement
travel_intervention = 'swap_networks' 

# options: 'none', 'swap_networks', 'inc_child_trav', 'inc_all_trav'
# none: no travel intervention, no child travelers
# swap_networks: increase proportion of travelers that are children to .15  and switch to holiday network
# inc_child_trav: increase proportion of travelers that are children to .15
# inc_all_trav: increase proportion of travelers that are children to .15 and rate of travel

# indicate which holiday timing to implement
timing = 'actual' 

# options: 'actual', 'extreme', 'late_holiday_3wk', 'late_holiday_6wk', 'late_holiday_9wk'
# actual: regular intervention timing
# extreme: extra long intervention duration
# late_holiday_3wk: shift intervention forward by 3 weeks
# late_holiday_6wk: shift intervention forward by 6 weeks
# late_holiday_9wk: shift intervention forward by 9 weeks