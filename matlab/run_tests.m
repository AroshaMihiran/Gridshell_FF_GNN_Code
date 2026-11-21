function reports=run_tests()
%RUN_TESTS Analytical, geometry, signed-load and timber-check regression suite.
root=setup_paths();
reports=struct('panel',test_panel_integration(),'boundary',test_boundary_panels(), ...
    'boundaryFrame',test_boundary_frame_handoff(),'frame',test_frame3d(), ...
    'frameHandoff',test_frame3d_handoff(),'memberChecks',test_timber_frame_checks(), ...
    'sequential',test_sequential_workflow(),'signedWind',test_signed_wind());
gs_write_json(fullfile(root,'outputs','tests','test_summary.json'),reports);
fprintf('All MATLAB test groups passed. Reports: outputs/tests\n');
end
