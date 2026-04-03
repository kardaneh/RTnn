Data Handling
=============

NetCDF Format
-------------

RTnn works with NetCDF files in the format: `rtnetcdf_XXX_YYYY.nc`

- `XXX`: Processor rank (3 digits)
- `YYYY`: Year (4 digits)

Data Structure
--------------

Input variables:
- coszang: Cosine of solar zenith angle
- laieff_collim: Collimated leaf area index
- laieff_isotrop: Isotropic leaf area index
- leaf_ssa: Leaf single scattering albedo
- leaf_psd: Leaf phase function asymmetry
- rs_surface_emu: Surface reflectance

Output variables:
- collim_alb: Collimated albedo
- collim_tran: Collimated transmittance
- isotrop_alb: Isotropic albedo
- isotrop_tran: Isotropic transmittance
