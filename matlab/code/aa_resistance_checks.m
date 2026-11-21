function checks=aa_resistance_checks(actions,section,lengths,material,options,checkMode)
%AA_RESISTANCE_CHECKS Change resistance inputs only; never change frame actions.
% The ablation omits M2 and M3 from the selected strength/buckling interactions.
% Axial sign, strength, member length, stiffness, buckling and all options stay.
checkMode=validatestring(checkMode,{'full','omit_bending'});
resistanceActions=actions;
if strcmp(checkMode,'omit_bending'),resistanceActions(:,5:6)=0;end
checks=timber_frame_checks(resistanceActions,section,lengths,material,options);
end
