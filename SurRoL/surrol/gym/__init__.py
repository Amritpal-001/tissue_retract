from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)

#TissueRetract_v0
register(
    id='TissueRetract-v0',
    entry_point='surrol.tasks.tissue_retract:TissueRetract_v0',
    max_episode_steps=50,
)

#TissueRetract_v1
register(
    id='TissueRetract-v1',
    entry_point='surrol.tasks.tissue_retract:TissueRetract_v1',
    max_episode_steps=50,
)

#TissueRetract_v2: 
register(
    id='TissueRetract-v2',
    entry_point='surrol.tasks.tissue_retract:TissueRetract_v2',
    max_episode_steps=50,
)

#TissueRetract_v3:
register(
    id='TissueRetract-v3',
    entry_point='surrol.tasks.tissue_retract:TissueRetract_v3',
    max_episode_steps=50,
)
