# ========== #
# Unit Tests #
# ========== #

enable_testing()

setup_test(test_eos)
setup_test(test_implicit_cfl)
setup_test(test_implicit_face_work_jacobian)
setup_test(test_vic_partial_tangential)
setup_test(test_radiating_boundary)
setup_test(test_plm)
setup_test(test_weno)
setup_test(test_x1_coeffs)
setup_test(test_x1_dim_guard)
setup_test(test_x1_stretched_refusal)
setup_test(test_x1_qmax_guard)
setup_test(test_hydro_ref_x1_rf_wall)
setup_test(test_uniform_module_tree)
setup_test(test_riemann)
if(CUDA)
  setup_test(test_weno5_cuda_line)
  setup_test(test_scalar_fused_weno5)
  setup_test(test_scalar_fused_carry)
  setup_test(test_scalar_fused_transfer)
endif()
setup_test(test_backward_substitution)
setup_test(test_lu_failure)
setup_test(test_scalar)
setup_test(test_rectify)
setup_test(test_flip_zero_count)
setup_test(test_cubed_sphere_exchange)
setup_test(test_coordinate)
setup_test(test_cp5_nghost)
setup_test(test_stretched_x1)
setup_test(test_deck_x1_wiring)
setup_test(test_stretched_x1_ref)
foreach(_arm "wb_ref4;SNAP_WB_REF4=1" "x1_centroid;SNAP_X1_CENTROID_EXACT=1")
  list(GET _arm 0 _name)
  list(GET _arm 1 _env)
  add_test(NAME test_stretched_x1_ref_${_name}.${buildl}
           COMMAND test_stretched_x1_ref.${buildl}
                   --gtest_filter=StretchedX1Ref.wb_ref4_switch_runs:StretchedX1Ref.balance_column_wb_ref4_runs)
  set_tests_properties(test_stretched_x1_ref_${_name}.${buildl}
                       PROPERTIES ENVIRONMENT "${_env}")
endforeach()
add_test(NAME test_stretched_x1_ref_radial_exact_off.${buildl}
         COMMAND test_stretched_x1_ref.${buildl}
                 --gtest_filter=StretchedX1Ref.refuses_curvature_with_radial_exact_off)
set_tests_properties(test_stretched_x1_ref_radial_exact_off.${buildl}
                     PROPERTIES ENVIRONMENT "SNAP_GRAVITY_WORK_RADIAL_EXACT=0")
add_test(NAME test_stretched_x1_ref_x1_mass_covariance.${buildl}
         COMMAND test_stretched_x1_ref.${buildl}
                 --gtest_filter=StretchedX1Ref.x1_mass_covariance_switch_runs)
set_tests_properties(
  test_stretched_x1_ref_x1_mass_covariance.${buildl}
  PROPERTIES ENVIRONMENT "SNAP_X1_MASS_COVARIANCE=1"
             PASS_REGULAR_EXPRESSION "\\[  PASSED  \\] 1 test\\.")
setup_test(test_radial_face_moments)
setup_test(test_hydrostatic)
setup_test(test_hydro_options)
setup_test(test_cycle_diagnostics)
setup_test(test_flux_positivity_carry)
# its limiter-off arm must run without the step 4.C donor limiter, which the
# face-flux covariance (on by default, #289) also switches on
set_tests_properties(test_flux_positivity_carry.${buildl}
                     PROPERTIES ENVIRONMENT "SNAP_FLUX_COVARIANCE=0")
setup_test(test_yaml_keys)

# Reads the example decks from the source tree; it needs no build and no snapy import.
add_test(
  NAME test_example_species_composition
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_example_species_composition.py
)
set_tests_properties(test_example_species_composition PROPERTIES
  TIMEOUT 60
  LABELS "examples;species"
)

# Reads patches/ from the source tree; the compile arm uses the build's own compiler.
add_test(
  NAME test_yamlcpp_cstdint_patch
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_yamlcpp_cstdint_patch.py
          --cxx ${CMAKE_CXX_COMPILER}
          --yaml-cpp-src ${yaml-cpp_SOURCE_DIR}
)
set_tests_properties(test_yamlcpp_cstdint_patch PROPERTIES
  TIMEOUT 120
  LABELS "build;patches"
)

setup_test(test_sedimentation_guards)
setup_test(test_balance_column)
setup_test(test_hydro_ref_x1)
add_test(NAME test_hydro_ref_x1_stretched_cuda.${buildl}
         COMMAND test_hydro_ref_x1.${buildl}
                 --gtest_filter=HydroRefX1Dispatch.cuda_matches_cpu_on_the_stretched_rows)
set_tests_properties(test_hydro_ref_x1_stretched_cuda.${buildl}
                     PROPERTIES TIMEOUT 120 LABELS "hydro;gpu")
setup_test(test_uranus_cycle1_abort)
setup_test(test_pref_local_seam)
# UCzhangxi/snapy#44: the same 4-vs-1 x1-block comparison (200 steps, 1e-12)
# with the x2/x3 face-flux covariance on; it must run and pass
add_test(NAME test_pref_local_seam_flux_covariance.${buildl}
         COMMAND test_pref_local_seam.${buildl})
set_tests_properties(
  test_pref_local_seam_flux_covariance.${buildl}
  PROPERTIES ENVIRONMENT
             "GTEST_FILTER=HydroRefX1.in_process_split_matches_one_block_after_200_steps;SNAP_FLUX_COVARIANCE=1"
             PASS_REGULAR_EXPRESSION "\\[  PASSED  \\] 1 test\\.")
setup_test(test_gravity_work_cache_to)
setup_test(test_x1_seam_split)
# one arm per entry: its environment, its one test, which must run and pass (a
# skip or an empty filter prints "PASSED ] 0 tests" and fails the entry)
function(seam_arm name env test)
  set(_n test_x1_seam_split${name}.${buildl})
  if(name)
    add_test(NAME ${_n} COMMAND test_x1_seam_split.${buildl})
  endif()
  set_tests_properties(
    ${_n} PROPERTIES ENVIRONMENT "GTEST_FILTER=X1SeamSplit.${test};${env}"
                     PASS_REGULAR_EXPRESSION "\\[  PASSED  \\] 1 test\\.")
endfunction()
seam_arm("" "" switches_off_nghost_1_sets_up)
seam_arm(_x1_centroid "SNAP_X1_CENTROID_EXACT=1"
         centroid_exact_split_matches_one_block)
seam_arm(_radial_exact "SNAP_GRAVITY_WORK_RADIAL_EXACT=1" radial_exact_split_gap)
# the control: the face form without the corrected work
seam_arm(_radial_exact_off "SNAP_GRAVITY_WORK_RADIAL_EXACT=0"
         radial_exact_split_gap)
# the VIC row's per-block slope relies on an implicit scheme refusing an x1 split
seam_arm(_vic_refuses_split "" vic_refuses_x1_split)
# SNAP_WB_REF4 alone: its resolution flag switching on next to the seam
seam_arm(_wb_ref4 "SNAP_WB_REF4=1"
         wb_ref4_flag_at_the_seam_split_matches_one_block)
# SNAP_WB_REF4 with grav1 = 0 on nghost 1: no reference, so no nghost check
seam_arm(_wb_ref4_gravity_0 "SNAP_WB_REF4=1" wb_ref4_gravity_0_nghost_1_steps)
# SNAP_X1_MASS_COVARIANCE refuses a block with fewer than 3 x1 cells
seam_arm(_x1_mass_covariance "SNAP_X1_MASS_COVARIANCE=1"
         x1_mass_covariance_needs_3_x1_cells)
if(CUDA)
  # the same arms on the GPU; without one the test skips and so does the entry
  seam_arm(_x1_centroid_cuda "SNAP_X1_CENTROID_EXACT=1"
           centroid_exact_split_matches_one_block_cuda)
  seam_arm(_radial_exact_cuda "SNAP_GRAVITY_WORK_RADIAL_EXACT=1"
           radial_exact_split_gap_cuda)
  seam_arm(_radial_exact_off_cuda "SNAP_GRAVITY_WORK_RADIAL_EXACT=0"
           radial_exact_split_gap_cuda)
  seam_arm(_wb_ref4_cuda "SNAP_WB_REF4=1"
           wb_ref4_flag_at_the_seam_split_matches_one_block_cuda)
  foreach(_a _x1_centroid_cuda _radial_exact_cuda _radial_exact_off_cuda
             _wb_ref4_cuda)
    set_tests_properties(
      test_x1_seam_split${_a}.${buildl}
      PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\] 1 test")
  endforeach()
endif()
# the same column with its two x1 blocks on two ranks; the arm is the env
setup_parallel_test(test_x1_seam_split_mp 2)
function(seam_arm_mp name env)
  set(_n test_x1_seam_split_mp${name}.${buildl})
  add_test(NAME ${_n} COMMAND torchrun --no-python --nproc-per-node=2
                              $<TARGET_FILE:test_x1_seam_split_mp.${buildl}>)
  set_tests_properties(${_n} PROPERTIES ENVIRONMENT "${env}")
endfunction()
seam_arm_mp(_x1_centroid "SNAP_X1_CENTROID_EXACT=1")
seam_arm_mp(_radial_exact "SNAP_GRAVITY_WORK_RADIAL_EXACT=1")
seam_arm_mp(_radial_exact_off "SNAP_GRAVITY_WORK_RADIAL_EXACT=0")
seam_arm_mp(_wb_ref4 "SNAP_WB_REF4=1")
setup_test(test_wb_wall_corner)
setup_test(test_face_floor)
setup_test(test_wb_ref_wall)
# #289: these two pin values of the kernel's x1 reference; run them again with
# SNAP_WB_REF4 set, where each checks the switched reference instead
foreach(_t test_balance_column test_face_floor)
  add_test(NAME ${_t}_wb_ref4.${buildl} COMMAND ${_t}.${buildl})
  set_tests_properties(${_t}_wb_ref4.${buildl} PROPERTIES ENVIRONMENT
                                                          "SNAP_WB_REF4=1")
  # SNAP_X1_CENTROID_EXACT alone implies SNAP_WB_REF4: the same checks hold
  add_test(NAME ${_t}_x1_centroid.${buildl} COMMAND ${_t}.${buildl})
  set_tests_properties(${_t}_x1_centroid.${buildl}
                       PROPERTIES ENVIRONMENT "SNAP_X1_CENTROID_EXACT=1")
endforeach()
setup_test(test_netcdf_utils)
setup_test(test_user_output)
setup_test(test_process_group)
setup_test(test_refine)
setup_test(test_diffusion)
setup_test(test_diffusion_cubed_sphere)
set_tests_properties(test_diffusion_cubed_sphere.${buildl} PROPERTIES TIMEOUT 120)
add_test(NAME test_diffusion_cubed_sphere_mp.${buildl}
  COMMAND torchrun --standalone --no-python --nproc-per-node=2
    $<TARGET_FILE:test_diffusion_cubed_sphere.${buildl}>
    --gtest_filter=CubedSphereDiffusion.constant_state_and_rigid_rotation:CubedSphereDiffusion.split_panels_preserve_constants_and_rotation:CubedSphereDiffusion.distributed_shared_heat_flux)
set_tests_properties(test_diffusion_cubed_sphere_mp.${buildl}
  PROPERTIES TIMEOUT 120 LABELS "parallel;forcing" ENVIRONMENT "BACKEND=gloo")
setup_test(test_diffusion_cubed_sphere_two_gpu)
set_tests_properties(test_diffusion_cubed_sphere_two_gpu.${buildl} PROPERTIES
  TIMEOUT 60
  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
  LABELS "forcing;gpu")
if(CUDA AND UCX_FOUND AND COMMUX_CUDA_FOUND)
  find_program(SNAPY_TWO_GPU_TORCHRUN_EXECUTABLE NAMES torchrun)
  if(SNAPY_TWO_GPU_TORCHRUN_EXECUTABLE)
    add_test(NAME test_diffusion_cubed_sphere_two_gpu_ucx.${buildl}
      COMMAND ${SNAPY_TWO_GPU_TORCHRUN_EXECUTABLE}
        --standalone --no-python --nproc-per-node=2
        $<TARGET_FILE:test_diffusion_cubed_sphere_two_gpu.${buildl}>)
    set_tests_properties(test_diffusion_cubed_sphere_two_gpu_ucx.${buildl}
      PROPERTIES TIMEOUT 60 PROCESSORS 2 LABELS "parallel;forcing;gpu;ucx"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
        ENVIRONMENT "BACKEND=ucx;DEVICE=cuda;UCX_TLS=self,tcp,cuda_copy,cuda_ipc")
  endif()
endif()
setup_test(test_diffusion_moist)
setup_test(test_diffusion_x1_scale)
setup_test(test_condensate_conservation)
setup_test(test_parentless_cloud)
setup_test(test_fix_vapor_volume)
setup_test(test_fix_vapor_deficit)
setup_test(test_check_redo_vapor_deficit)
setup_test(test_species_face_clamp)
setup_test(test_parentless_cloud_nb1)
setup_test(test_vapor_column_nb1)
setup_test(test_wall_saturation)
setup_test(test_cloud_parent_slots)
setup_test(test_two_cards_species)
setup_test(test_forcing)
setup_parallel_test(test_exchange 2)
setup_parallel_test(test_output_barrier 2)
# The Gloo device must honor the selected interface rather than resolve the host.
add_test(NAME test_gloo_invalid_interface.${buildl}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_gloo_invalid_interface.py
    torchrun --local-addr=127.0.0.1 --no-python --nproc-per-node=2
    $<TARGET_FILE:test_output_barrier.${buildl}>)
set_tests_properties(test_gloo_invalid_interface.${buildl}
  PROPERTIES ENVIRONMENT "BACKEND=gloo;GLOO_SOCKET_IFNAME=snapy_no_such_interface"
    TIMEOUT 60)
setup_parallel_test(test_check_redo_parallel 2)
setup_parallel_test(test_cycle_diagnostics_parallel 2)
if(APPLE)  # Gloo is the Mac backend; Linux runs UCX
  setup_parallel_test(test_gloo_one_tensor 2)
endif()
setup_parallel_test(test_sedimentation_cubed_seam 2)
# The same two ranks on Gloo, which Linux CI (UCX by default) would not reach:
# the limiter exchanges more than one variable, and a Gloo send takes one
# tensor. With UCX built, the Gloo result must equal the UCX one bit for bit.
if(UCX_FOUND)
  add_test(NAME test_sedimentation_cubed_seam_gloo
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/run_seam_backends.py
                   --exe $<TARGET_FILE:test_sedimentation_cubed_seam.${buildl}>
                   --compare-ucx)
  set_tests_properties(test_sedimentation_cubed_seam_gloo PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    SKIP_RETURN_CODE 125
    LABELS "exchange")
endif()

setup_parallel_test(test_parentless_cloud_nb1_mp 2)
if(UCX_FOUND)  # the same two ranks on Gloo, which Linux (UCX) would not reach
  add_test(NAME test_parentless_cloud_nb1_mp_gloo
           COMMAND torchrun --no-python --nproc-per-node=2
                   $<TARGET_FILE:test_parentless_cloud_nb1_mp.${buildl}>)
  set_tests_properties(test_parentless_cloud_nb1_mp_gloo PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ENVIRONMENT "BACKEND=gloo")
endif()

if(UCX_FOUND)
  add_test(
    NAME test_exchange_ucx
    COMMAND torchrun
            --no-python
            --nproc-per-node=2
            $<TARGET_FILE:test_exchange.${buildl}>
  )
  set_tests_properties(test_exchange_ucx PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ENVIRONMENT
      "BACKEND=ucx;BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
    LABELS "ucx;exchange"
  )

  if(CUDA)
    add_test(
      NAME test_exchange_ucx_cuda
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_backend.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type ${buildl}
              --backend ucx
              --device cuda
    )
    set_tests_properties(test_exchange_ucx_cuda PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      SKIP_RETURN_CODE 125
      LABELS "ucx;exchange;gpu"
    )
  endif()
endif()
set_tests_properties(test_exchange.release PROPERTIES
  ENVIRONMENT
    "BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
)

add_test(
  NAME test_radiating_boundary_python
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_radiating_boundary_python.py
)
set_tests_properties(test_radiating_boundary_python PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  LABELS "python;boundary"
)
set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS test_radiating_boundary_python)

# Every python ctest that imports snapy goes through this, so it is enrolled for
# SNAPY_TEST_PYTHONPATH below; the raw add_test ones drive a built binary and import none.
function(snapy_add_python_test name)
  cmake_parse_arguments(ARG "" "TIMEOUT;SCRIPT" "LABELS;ARGS" ${ARGN})
  if(NOT ARG_SCRIPT)
    set(ARG_SCRIPT ${name}.py)
  endif()
  add_test(
    NAME ${name}_python
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SCRIPT} ${ARG_ARGS}
  )
  set_tests_properties(${name}_python PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    LABELS "${ARG_LABELS}"
  )
  if(ARG_TIMEOUT)
    set_tests_properties(${name}_python PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS ${name}_python)
endfunction()

snapy_add_python_test(test_mesh_exchange LABELS "python;exchange")
snapy_add_python_test(test_cubed_sphere_vertical_velocity_exchange
                      LABELS "python;exchange;cubed-sphere;diagnostic")
snapy_add_python_test(test_jit_user_forcing TIMEOUT 30 LABELS "python;forcing")
snapy_add_python_test(test_fix_vapor_reports_failure TIMEOUT 60 LABELS "python;eos;positivity")
snapy_add_python_test(test_fix_vapor_stale_count TIMEOUT 60 LABELS "python;eos;positivity;redo")
snapy_add_python_test(test_fix_vapor_counts_every_column TIMEOUT 120 LABELS "python;eos;positivity")
snapy_add_python_test(test_implicit_advection_cfl TIMEOUT 60 LABELS "python;hydro;implicit")
snapy_add_python_test(test_implicit_gravity_tall_column TIMEOUT 120 LABELS "python;hydro;implicit;gravity")
# the same ladder on one cubed-sphere face block, where the corrected-PE work carries the
# centroid term (#300)
snapy_add_python_test(test_implicit_gravity_tall_column_cubed SCRIPT test_implicit_gravity_tall_column.py
                      ARGS --geometry gnomonic-equiangle TIMEOUT 120
                      LABELS "python;hydro;implicit;gravity;cubed-sphere")
# #299: the runner above fails when a ladder child dies or prints an incomplete ladder
snapy_add_python_test(test_tall_column_runner_child_failure TIMEOUT 120 LABELS "python;hydro;implicit")
snapy_add_python_test(test_implicit_face_work_operator TIMEOUT 300 LABELS "python;hydro;implicit;gravity")
snapy_add_python_test(test_implicit_stratified_solid TIMEOUT 300 LABELS "python;hydro;implicit;gravity")
# the same two tests with the corrected-PE work on (#296): they detect the switch and
# measure E + P; the tall column runs both arms itself
foreach(_t test_implicit_face_work_operator test_implicit_stratified_solid)
  snapy_add_python_test(${_t}_radial_exact SCRIPT ${_t}.py TIMEOUT 300
                        LABELS "python;hydro;implicit;gravity")
  set_tests_properties(${_t}_radial_exact_python PROPERTIES ENVIRONMENT
                       "SNAP_GRAVITY_WORK_RADIAL_EXACT=1")
endforeach()
snapy_add_python_test(test_gravity_work_fixer TIMEOUT 120 LABELS "python;hydro;gravity;energy")
snapy_add_python_test(test_gravity_work_radial_exact TIMEOUT 300 LABELS "python;hydro;gravity")
snapy_add_python_test(test_implicit_options_type TIMEOUT 60 LABELS "python;implicit")
snapy_add_python_test(test_shear_cfl TIMEOUT 60 LABELS "python;hydro;implicit")
snapy_add_python_test(test_check_redo_floor TIMEOUT 60 LABELS "python;mesh;redo")
snapy_add_python_test(test_check_redo_saturation TIMEOUT 60 LABELS "python;mesh;redo;eos")
snapy_add_python_test(test_eos_temp2inteng TIMEOUT 60 LABELS "python;eos")
snapy_add_python_test(test_flux_positivity TIMEOUT 300 LABELS "python;positivity")
# the base arm is the unlimited scheme and must go negative; with the face-flux
# covariance on (the default, #289) the donor limiter runs in every arm
set_tests_properties(test_flux_positivity_python
                     PROPERTIES ENVIRONMENT "SNAP_FLUX_COVARIANCE=0")
snapy_add_python_test(test_flux_positivity_cubedsphere TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
# with the covariance on (the default) the donor limiter runs in the base arm
# too; this arm keeps the unlimited base arm covered
snapy_add_python_test(test_flux_positivity_cubedsphere_cov_off
                      SCRIPT test_flux_positivity_cubedsphere.py TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
set_tests_properties(test_flux_positivity_cubedsphere_cov_off_python
                     PROPERTIES ENVIRONMENT "SNAP_FLUX_COVARIANCE=0")
# UCzhangxi/snapy#44 (B1): the six-panel scalar seam test with the x2/x3
# face-flux covariance on, whose x1 differences read the cross-panel face
# states' x1 ghost rows next to the radial walls
snapy_add_python_test(test_flux_positivity_cubedsphere_flux_covariance
                      SCRIPT test_flux_positivity_cubedsphere.py TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
set_tests_properties(test_flux_positivity_cubedsphere_flux_covariance_python
                     PROPERTIES ENVIRONMENT "SNAP_FLUX_COVARIANCE=1")
snapy_add_python_test(test_flux_positivity_cubedsphere_moist TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
snapy_add_python_test(test_roe_cubed_sphere_rest TIMEOUT 120
                      LABELS "python;riemann;cubed-sphere")
snapy_add_python_test(test_cubed_sphere_cell_volume TIMEOUT 120
                      LABELS "python;coordinate;cubed-sphere")
snapy_add_python_test(test_stage2_cubed_stretched ARGS --device cpu TIMEOUT 3600
                      LABELS "python;coordinate;cubed-sphere;stage2")
if(CUDA)
  snapy_add_python_test(test_stage2_cubed_stretched_cuda SCRIPT test_stage2_cubed_stretched.py
                        ARGS --device cuda TIMEOUT 3600 LABELS "python;coordinate;cubed-sphere;stage2")
endif()
snapy_add_python_test(test_outflow_stale_ghosts TIMEOUT 120
                      LABELS "python;boundary;implicit;cubed-sphere")
snapy_add_python_test(test_tracer_dry_convention TIMEOUT 120 LABELS "python;scalar;implicit;forcing")
snapy_add_python_test(test_eos_species_registry TIMEOUT 120 LABELS "python;eos")
snapy_add_python_test(test_stage_forcing_dry_tracer TIMEOUT 60 LABELS "python;forcing;scalar")
snapy_add_python_test(test_forcing_species_clip TIMEOUT 120 LABELS "python;forcing;positivity")
snapy_add_python_test(test_dry_carry_zero_base TIMEOUT 60 LABELS "python;forcing;scalar")
snapy_add_python_test(test_local_horizontal_cells TIMEOUT 60 LABELS "python;mesh;coordinate")
snapy_add_python_test(test_diffusion_x1_scale TIMEOUT 120 LABELS "python;forcing")
snapy_add_python_test(test_horizontal_flux_covariance TIMEOUT 180 LABELS "python;hydro")
snapy_add_python_test(test_flux_covariance_rows TIMEOUT 300
                      LABELS "python;hydro;cubed-sphere")
snapy_add_python_test(test_flux_covariance_seams TIMEOUT 300
                      LABELS "python;hydro;cubed-sphere")
# which flux rows the covariance changes: Cartesian vs curved, dry vs moist
snapy_add_python_test(test_flux_covariance_rowset TIMEOUT 300 LABELS "python;hydro")
snapy_add_python_test(test_wb_ref4_order TIMEOUT 300 LABELS "python;hydro")
snapy_add_python_test(test_x1_centroid_rest TIMEOUT 300 LABELS "python;hydro")
snapy_add_python_test(test_radial_order TIMEOUT 300 LABELS "python;hydro")
snapy_add_python_test(test_radial_order_x1_centroid SCRIPT test_radial_order.py
                      ARGS --x1-centroid TIMEOUT 300 LABELS "python;hydro")
# #289: a passive scalar and the same tracer as a hydro species, covariance on and off
snapy_add_python_test(test_scalar_flux_covariance TIMEOUT 300 LABELS "python;hydro;scalar")
snapy_add_python_test(test_scalar_stratified TIMEOUT 300 LABELS "python;scalar")
# upper-bound must not drive a donor negative
snapy_add_python_test(test_scalar_upper_bound_donor TIMEOUT 300 LABELS "python;scalar")
# tracer and species ratios under the species positivity limiter (#338)
snapy_add_python_test(test_tracer_ratio TIMEOUT 120 LABELS "python;hydro;scalar;positivity")
# #289: the same, on a column split in x1, four blocks in one process vs one
snapy_add_python_test(test_scalar_cov_x1_split TIMEOUT 300 LABELS "python;hydro;scalar")
set_tests_properties(test_scalar_cov_x1_split_python
                     PROPERTIES ENVIRONMENT "SNAP_FLUX_COVARIANCE=1")
# #298: the x1 stencil caches on blocks with too few x1 cells
snapy_add_python_test(test_stencil_cache_small_block TIMEOUT 300 LABELS "python;hydro")
# these check the plain face work (E+PE_d, or values recorded with it); the
# corrected work, on by default with gravity-work: face, conserves E+P instead
set_tests_properties(
  test_forcing.${buildl} test_gravity_work_fixer_python
  test_horizontal_flux_covariance_python test_flux_covariance_rows_python
  PROPERTIES ENVIRONMENT "SNAP_GRAVITY_WORK_RADIAL_EXACT=0")
# their switch-off arms; the _radial_exact entries run them with it on
set_tests_properties(
  test_implicit_face_work_operator_python test_implicit_stratified_solid_python
  PROPERTIES ENVIRONMENT "SNAP_GRAVITY_WORK_RADIAL_EXACT=0")

# CUDA variants reuse the same scripts and assertions.
if(CUDA)
  snapy_add_python_test(test_fix_vapor_reports_failure_cuda
    SCRIPT test_fix_vapor_reports_failure.py ARGS --device cuda
    TIMEOUT 60 LABELS "python;eos;positivity;cuda")
  snapy_add_python_test(test_fix_vapor_stale_count_cuda
    SCRIPT test_fix_vapor_stale_count.py ARGS --device cuda
    TIMEOUT 60 LABELS "python;eos;positivity;redo;cuda")
  snapy_add_python_test(test_check_redo_saturation_cuda
    SCRIPT test_check_redo_saturation.py ARGS --device cuda
    TIMEOUT 60 LABELS "python;mesh;redo;eos;cuda")
  snapy_add_python_test(test_flux_positivity_cubedsphere_cuda
    SCRIPT test_flux_positivity_cubedsphere.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;positivity;cubed-sphere;cuda")
  snapy_add_python_test(test_flux_positivity_cubedsphere_moist_cuda
    SCRIPT test_flux_positivity_cubedsphere_moist.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;positivity;cubed-sphere;cuda")
  snapy_add_python_test(test_implicit_face_work_operator_cuda SCRIPT test_implicit_face_work_operator.py
    ARGS --device cuda TIMEOUT 300 LABELS "python;hydro;implicit;gravity;cuda")
  snapy_add_python_test(test_vic_moist_device
    TIMEOUT 120 LABELS "python;implicit;cuda")
  snapy_add_python_test(test_wb_ref4_order_cuda
    SCRIPT test_wb_ref4_order.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;hydro;cuda")
  snapy_add_python_test(test_x1_centroid_rest_cuda
    SCRIPT test_x1_centroid_rest.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;hydro;cuda")
  snapy_add_python_test(test_radial_order_cuda
    SCRIPT test_radial_order.py ARGS --device cuda TIMEOUT 300 LABELS "python;hydro;cuda")
  snapy_add_python_test(test_radial_order_x1_centroid_cuda SCRIPT test_radial_order.py
    ARGS --device cuda --x1-centroid TIMEOUT 300 LABELS "python;hydro;cuda")
  snapy_add_python_test(test_gravity_work_radial_exact_cuda
    SCRIPT test_gravity_work_radial_exact.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;hydro;gravity;cuda")
  snapy_add_python_test(test_scalar_flux_covariance_cuda
    SCRIPT test_scalar_flux_covariance.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;hydro;scalar;cuda")
  snapy_add_python_test(test_scalar_upper_bound_donor_cuda
    SCRIPT test_scalar_upper_bound_donor.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;scalar;cuda")
  snapy_add_python_test(test_tracer_ratio_cuda
    SCRIPT test_tracer_ratio.py ARGS --device cuda
    TIMEOUT 120 LABELS "python;hydro;scalar;positivity;cuda")
  snapy_add_python_test(test_tracer_dry_convention_cuda
    SCRIPT test_tracer_dry_convention.py ARGS --device cuda
    TIMEOUT 120 LABELS "python;scalar;implicit;forcing;cuda")
  # the fused passive-scalar path (SNAP_SCALAR_FUSED=1) against the default path: the flow and the PLM case
  # bit for bit, the WENO5 cases to round-off
  snapy_add_python_test(test_scalar_fused
    TIMEOUT 600 LABELS "python;scalar;cuda")
  # the fused path on a column split in x1, with the covariance (its lateral
  # flux keeps the default path there)
  snapy_add_python_test(test_scalar_cov_x1_split_fused_cuda
    SCRIPT test_scalar_cov_x1_split.py ARGS --device cuda
    TIMEOUT 300 LABELS "python;hydro;scalar;cuda")
  set_tests_properties(test_scalar_cov_x1_split_fused_cuda_python PROPERTIES
                       ENVIRONMENT "SNAP_FLUX_COVARIANCE=1;SNAP_SCALAR_FUSED=1")
  set_tests_properties(
    test_fix_vapor_reports_failure_cuda_python
    test_fix_vapor_stale_count_cuda_python
    test_check_redo_saturation_cuda_python
    test_flux_positivity_cubedsphere_cuda_python
    test_flux_positivity_cubedsphere_moist_cuda_python
    test_vic_moist_device_python
    test_wb_ref4_order_cuda_python
    test_x1_centroid_rest_cuda_python
    test_radial_order_cuda_python
    test_radial_order_x1_centroid_cuda_python
    test_gravity_work_radial_exact_cuda_python
    test_scalar_flux_covariance_cuda_python
    test_scalar_upper_bound_donor_cuda_python
    test_tracer_ratio_cuda_python
    test_tracer_dry_convention_cuda_python
    test_scalar_fused_python
    test_scalar_cov_x1_split_fused_cuda_python
    PROPERTIES SKIP_RETURN_CODE 125)
endif()

# Checks the SNAPY_TEST_PYTHONPATH wiring below. Its own PYTHONPATH stands in for the
# caller's and must survive behind SNAPY_TEST_PYTHONPATH.
set(_marker "${CMAKE_CURRENT_BINARY_DIR}/caller_pythonpath_marker")
if(SNAPY_TEST_PYTHONPATH)
  set(_import_path_args "${SNAPY_TEST_PYTHONPATH}" "${_marker}")
endif()
snapy_add_python_test(test_python_import_path LABELS "python" ARGS ${_import_path_args})
set_tests_properties(test_python_import_path_python PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_marker}"
  SKIP_RETURN_CODE 125
)

snapy_add_python_test(test_example_ic_stretched TIMEOUT 120 LABELS "python;examples"
  ARGS --exe ${CMAKE_BINARY_DIR}/bin/run_hydro.${buildl})
set_tests_properties(test_example_ic_stretched_python PROPERTIES
                     SKIP_RETURN_CODE 125)

file(GLOB inputs *.yaml *.py)
# create tests folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
foreach(input ${inputs})
  get_filename_component(input_name ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${input_name})
endforeach()

if (DEFINED KINTERA_DATA_DIR AND EXISTS "${KINTERA_DATA_DIR}/nasa9.dat")
  execute_process(
    COMMAND ln -sf ${KINTERA_DATA_DIR}/nasa9.dat
            ${CMAKE_BINARY_DIR}/tests/nasa9.dat)
endif()

# ---- Python ctests: say which snapy they test -------------------------------------------
# `COMMAND ${Python3_EXECUTABLE} <test>.py` imports `snapy` through sys.path, so with nothing
# set these exercise whatever snapy is installed, not the tree they were configured from --
# a false red when the branch is fine and the install is stale, and a false green when the
# branch is broken and the install is fine. cmake builds only the C++ libraries here (the
# importable package comes from `pip install`), so there is nothing in the build tree to point
# at by default; give the directory that holds this build's installed snapy:
#     cmake -B build -DSNAPY_TEST_PYTHONPATH=/path/to/site-packages
set(SNAPY_TEST_PYTHONPATH "" CACHE PATH
    "PYTHONPATH for the *_python ctests; empty means they test the installed snapy")
get_property(SNAPY_PYTHON_TESTS GLOBAL PROPERTY SNAPY_PYTHON_TESTS)
if(SNAPY_TEST_PYTHONPATH)
  # Prepend, do not replace: the caller's PYTHONPATH may be what selects the kintera/pyharp
  # this tree was configured against, and a test's own ENVIRONMENT must survive.
  set_tests_properties(${SNAPY_PYTHON_TESTS} PROPERTIES ENVIRONMENT_MODIFICATION
                       "PYTHONPATH=path_list_prepend:${SNAPY_TEST_PYTHONPATH}")
  message(STATUS "python ctests import snapy from ${SNAPY_TEST_PYTHONPATH}")
else()
  message(WARNING
    "SNAPY_TEST_PYTHONPATH is unset: the *_python ctests will import the installed snapy, "
    "not this tree. Their verdicts do not constrain this branch.")
endif()

# ============= #
# Example Tests #
# ============= #
set(reference_examples straka)
if(FULL_TESTS)
  list(APPEND reference_examples shallow_xy shallow_splash)
endif()
foreach(example IN LISTS reference_examples)
  add_test(NAME test_${example}
           COMMAND ${CMAKE_COMMAND} -Dbuildl=release
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/run_${example}.cmake)
  set_tests_properties(test_${example} PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    LABELS "reference")
endforeach()

# straka at cfl 1.6 needs redos: a non-finite step must reach check_redo
# (gravity-work-fixer on), not stop the run
add_test(NAME test_straka_redo
         COMMAND ${CMAKE_COMMAND} -Dbuildl=release
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/run_straka_redo.cmake)
set_tests_properties(test_straka_redo PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  TIMEOUT 600)

add_test(
  NAME test_bryan_balance_ic
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_bryan_balance_ic.cmake
)

# three bryan runs, 6.4-6.8 s on one 16-core CPU node: three times that
set_tests_properties(test_bryan_balance_ic PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  TIMEOUT 21
)

if (NOT APPLE)
    add_test(
      NAME test_restart_multiblock
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_multiblock.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_multiblock PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      LABELS "restart;examples"
    )

    set(restart_tests
      cycle_limit
      dt_change
      output_schedule
      key_collision
      inplace_reorder)
    foreach(restart_test IN LISTS restart_tests)
      add_test(
        NAME test_restart_${restart_test}
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_${restart_test}.py
                --build-dir ${CMAKE_BINARY_DIR}
                --build-type release
      )
      set_tests_properties(test_restart_${restart_test} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        TIMEOUT 600
        SKIP_RETURN_CODE 125
        LABELS "restart;examples"
      )
    endforeach()

    add_test(
      NAME test_netcdf_axis_attrs
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/test_netcdf_axis_attrs.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )
    set_tests_properties(test_netcdf_axis_attrs PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 300
      SKIP_RETURN_CODE 125
      LABELS "output;examples"
    )

    add_test(
      NAME test_restart_resume_schedule
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/test_restart_resume_schedule.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )
    set_tests_properties(test_restart_resume_schedule PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      SKIP_RETURN_CODE 125
      LABELS "restart;examples"
    )
endif()

setup_parallel_test(test_mesh_multi_block 2)

if(NOT APPLE OR FULL_TESTS)
  add_test(
    NAME test_exchange_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_exchange_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )
endif()

if(FULL_TESTS)
  add_test(
    NAME test_shallow_splash_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend gloo
  )

  set_tests_properties(test_shallow_splash_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_xy_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_xy_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_shallow_xy_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 4
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_splash_ucx_cuda_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend ucx
            --device cuda
  )

  set_tests_properties(test_shallow_splash_ucx_cuda_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    SKIP_RETURN_CODE 125
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "gpu;ucx;long"
  )
endif()

# A build without CUDA skips every CUDA test, even when a GPU is visible.
# Registration is the gate; torch.cuda.is_available() is not.
if(NOT CUDA)
  get_property(_snapy_tests DIRECTORY PROPERTY TESTS)
  foreach(_t IN LISTS _snapy_tests)
    set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                 "SNAPY_BUILD_CUDA=set:0")
    set(_disable FALSE)
    if(_t MATCHES "cuda")
      set(_disable TRUE)
    endif()
    get_property(_labels TEST ${_t} PROPERTY LABELS)
    foreach(_l IN LISTS _labels)
      if(_l STREQUAL "cuda" OR _l STREQUAL "gpu")
        set(_disable TRUE)
      endif()
    endforeach()
    if(_disable)
      set_tests_properties(${_t} PROPERTIES DISABLED TRUE)
    endif()
  endforeach()
endif()

add_executable(g3_decks.${buildl} EXCLUDE_FROM_ALL g3_decks.cpp)
set_target_properties(g3_decks.${buildl}
                      PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})
target_include_directories(
  g3_decks.${buildl}
  PRIVATE ${CMAKE_BINARY_DIR}
          ${KINTERA_INCLUDE_DIR}
          ${SNAP_INCLUDE_DIR}
          ${HARP_INCLUDE_DIR}
          ${NETCDF_INCLUDES}
          ${TORCH_INCLUDE_DIR}
          ${TORCH_API_INCLUDE_DIR})
target_link_libraries(g3_decks.${buildl}
  PRIVATE snapy::snap
          $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>)
if (UNIX AND NOT APPLE)
  target_link_options(g3_decks.${buildl} PRIVATE -Wl,--no-as-needed)
endif()
