"""Timoshenko bending matrix derived independently from end-rotation flexibility."""
import numpy as np

def bending_by_flexibility(EI, GA, L):
    # Basic deformations are each end rotation relative to the chord.
    # Bending flexibility integrates the linear moment diagram; shear adds
    # the chord rotation associated with the common transverse shear force.
    flex = L/(6*EI)*np.array([[2., -1.], [-1., 2.]])
    flex += np.ones((2, 2))/(GA*L)
    compatibility = np.array([[1/L, 1., -1/L, 0.], [1/L, 0., -1/L, 1.]])
    return compatibility.T @ np.linalg.solve(flex, compatibility)

