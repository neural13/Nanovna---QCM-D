# Nanovna---QCM-D
This repository contains the Python code for controlling a NanoVNA-H, turning it into a quartz crystal microbalance with dissipation monitoring. The created interface allows the user to adjust the scanning frequency range and the number of points per scan. It also allows obtaining the resonance frequency and dissipation (raw data and fitted with the Lorentz function) as a function of time. It is also possible to view and obtain the conductance vs frequency curve for each scan cycle. The data obtained from each experiment can be automatically saved to a directory previously chosen by the user within the code. This code is derived from https://github.com/ttrftech/NanoVNA

Important! For the interface to function correctly, replace the text "***************your directory path here******************" in the lines 222 and 240  with the address of the directory where you want to save the files on your computer.

neural13. (2026). neural13/Nanovna---QCM-D: NanoVNA-QCM-D_2 (NanoVNA-QCM-D_2). Zenodo. https://doi.org/10.5281/zenodo.19439995
