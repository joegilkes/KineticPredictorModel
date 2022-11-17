from openbabel import openbabel as ob

def get_radical_state(obatom):
    '''Gets the radical state of a given OBAtom.'''
    typical_val = ob.GetTypicalValence(
        obatom.GetAtomicNum(), 
        obatom.GetTotalValence(),
        obatom.GetFormalCharge()
    )
    current_val = obatom.GetTotalValence()
    return typical_val - current_val

def fix_radicals(obmol):
    '''Standardise radical molecules within OpenBabel.
    
    OpenBabel sometimes struggles to parse structures with neighbouring
    radicals consistently, leading to multiple interpretations of radical
    structure coming from very similar geometries.

    This tries to fix the issue by enforcing that all neighbouring radicals
    should join together to form a bond, i.e. by transforming all radicals
    into their most stable state.

    Example:
        * The species [CH2][C]C and C[C]=C have equal likelihood of being
        detected from the same geometry (via `pybel.readfile()`). These
        species have the same atoms, but differ in their bonding with the
        former having a radical CH2 and a diradical C, and the latter
        having only a radical C. This function resolves the discrepancy
        by forming a bond between the neighbouring radical carbons in the
        former, transforming it into the latter and thus standardising
        the radical structure.
    '''
    bonds_changed = True
    while bonds_changed:
        bonds_changed = False
        for bond in ob.OBMolBondIter(obmol.OBMol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            a1_type = a1.GetType()
            a2_type = a2.GetType()
            if a1_type == 'H' or a2_type == 'H':
                continue
            a1_free_elecs = get_radical_state(a1)
            a2_free_elecs = get_radical_state(a2)
            if a1_free_elecs > 0 and a2_free_elecs > 0:
                bond.SetBondOrder(bond.GetBondOrder()+1)
                bonds_changed = True