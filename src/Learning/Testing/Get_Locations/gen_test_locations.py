from .cube_locations import gather_cube_test_positions
from .distr_locations import gather_distr_test_positions
from .shape_locations import gather_shape_test_positions

def gather_valid_test_positions(
    scene_type: str,
    file_name: str,
    boundary_restiction: str,
    n_episodes: int,
    n_steps: int,
    bot_type: str,
    max_deviation: float,
    always_maxDev: float,
    trj_type: str,
    distance_cubeReached: float,
    n_distractors: int = 3,
    shape = 'cube'
):
    if scene_type == 'cube':
        gather_cube_test_positions(
            file_name,
            boundary_restiction,
            n_episodes,
            n_steps,
            bot_type,
            max_deviation,
            always_maxDev,
            trj_type,
            distance_cubeReached
        )

    elif scene_type == 'distractor':
        gather_distr_test_positions(
            file_name,
            boundary_restiction,
            n_episodes,
            n_steps,
            bot_type,
            max_deviation,
            always_maxDev,
            trj_type,
            distance_cubeReached,
            n_distractors
        )

    elif scene_type == 'shape':
        gather_shape_test_positions(
            file_name,
            boundary_restiction,
            n_episodes,
            n_steps,
            bot_type,
            max_deviation,
            always_maxDev,
            trj_type,
            distance_cubeReached,
            shape
        )
    