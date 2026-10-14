import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors, Lipinski, Crippen
from rdkit.Chem.AtomPairs import Pairs, Torsions
from mordred import Calculator, descriptors
import warnings
import math

warnings.filterwarnings('ignore')


class MolecularDescriptorCalculator:
    """
    Calculate six types of molecular descriptors/fingerprints
    plus domain-specific descriptors:
    1. Mordred2D
    2. Mordred2D,3D
    3. ECFP4 (1024 bits)
    4. Avalon fingerprints (512 bits)
    5. MACCS keys (166 bits)
    6. RDKit fingerprints (1024 bits)
    7. Domain-specific descriptors (functional groups, volatility, logKoa-related)
    """

    def __init__(self):
        # Initialize Mordred calculators
        self.calc_2d = Calculator(descriptors, ignore_3D=True)
        self.calc_2d_3d = Calculator(descriptors, ignore_3D=False)

        # SMARTS patterns for functional groups
        self.functional_group_smarts = {
            'Carboxyl_COOH': 'C(=O)[OH]',
            'Hydroxyl_OH': '[OX2H]',
            'Nitro_NO2': '[NX3](=O)=O',
            'Carbonyl_C=O': '[CX3]=[OX1]',
            'Amino_NH2': '[NX3;H2]',
            'Ester_COOR': 'C(=O)[OX2]C',
            'Ether_ROR': '[OD2]([#6])[#6]',
            'Aldehyde_CHO': '[CX3H1](=O)[#6]',
            'Amide_CONH': 'C(=O)[NX3]',
            'Aromatic_ring': 'a',
        }

    def generate_3d_structure(self, mol):
        """Generate and optimize a 3D structure using RDKit."""
        if mol is None:
            return None
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            mol = Chem.RemoveHs(mol)
            return mol
        except:
            return None

    # ---------- 1. Core descriptor calculations ----------
    def calculate_mordred_2d(self, mol):
        if mol is None:
            return None
        try:
            df_desc = self.calc_2d(mol)
            desc_dict = df_desc.fill_missing().asdict()
            return desc_dict
        except:
            return None

    def calculate_mordred_2d_3d(self, mol_3d):
        if mol_3d is None:
            return None
        try:
            df_desc = self.calc_2d_3d(mol_3d)
            desc_dict = df_desc.fill_missing().asdict()
            return desc_dict
        except:
            return None

    def calculate_ecfp4(self, mol, n_bits=1024):
        if mol is None:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fp_list = list(fp)
            fp_dict = {f'ECFP4_{i}': bit for i, bit in enumerate(fp_list)}
            return fp_dict
        except:
            return None

    def calculate_avalon_fingerprint(self, mol, n_bits=512):
        if mol is None:
            return None
        try:
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            fp_list = list(fp)
            fp_dict = {f'AFP_{i}': bit for i, bit in enumerate(fp_list)}
            return fp_dict
        except ImportError:
            print("Please install avalon-toolkit: pip install avalon-toolkit")
            return None
        except:
            return None

    def calculate_maccs_keys(self, mol):
        if mol is None:
            return None
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_list = list(fp)[1:]  # skip the first bit (always 1)
            fp_dict = {f'MACCS_{i}': bit for i, bit in enumerate(fp_list)}
            return fp_dict
        except:
            return None

    def calculate_rdkit_fingerprint(self, mol, n_bits=1024):
        if mol is None:
            return None
        try:
            from rdkit.Chem import RDKFingerprint
            fp = RDKFingerprint(mol, fpSize=n_bits)
            fp_list = list(fp)
            fp_dict = {f'RDKFP_{i}': bit for i, bit in enumerate(fp_list)}
            return fp_dict
        except:
            return None

    # ---------- 2. Domain-specific descriptors ----------
    def calculate_domain_specific_descriptors(self, mol):
        """Calculate domain-specific descriptors: functional groups, volatility, logKoa-related."""
        if mol is None:
            return None
        try:
            results = {}

            # 1. Oxidation state and functional group counts
            results['Num_O_atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
            results['Num_N_atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            results['Num_C_atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            results['Num_H_atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)

            if results['Num_C_atoms'] > 0:
                results['O_to_C_ratio'] = results['Num_O_atoms'] / results['Num_C_atoms']
                results['N_to_C_ratio'] = results['Num_N_atoms'] / results['Num_C_atoms']
            else:
                results['O_to_C_ratio'] = 0
                results['N_to_C_ratio'] = 0

            for group_name, smarts in self.functional_group_smarts.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    results[f'FG_{group_name}_count'] = len(matches)
                else:
                    results[f'FG_{group_name}_count'] = 0

            results['Num_H_Donors'] = Lipinski.NumHDonors(mol)
            results['Num_H_Acceptors'] = Lipinski.NumHAcceptors(mol)
            results['Num_RotatableBonds'] = Lipinski.NumRotatableBonds(mol)
            results['Num_AromaticRings'] = Lipinski.NumAromaticRings(mol)
            results['Num_AliphaticRings'] = Lipinski.NumAliphaticRings(mol)

            # 2. Octanol-air partition coefficient related parameters
            results['MolLogP'] = Crippen.MolLogP(mol)
            results['MolMR'] = Crippen.MolMR(mol)
            results['TPSA'] = Descriptors.TPSA(mol)
            results['MolecularWeight'] = Descriptors.MolWt(mol)
            results['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
            results['FractionCSP3'] = Lipinski.FractionCSP3(mol)

            # 3. Volatility parameters
            results['BoilingPoint_est'] = self.estimate_boiling_point(mol)
            results['Volatility_index'] = self.calculate_volatility_index(mol)
            results['RingCount'] = Lipinski.RingCount(mol)
            results['Complexity'] = rdMolDescriptors.CalcNumAtomStereoCenters(mol) + rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

            return results
        except Exception as e:
            print(f"Error computing domain-specific descriptors: {str(e)}")
            return None

    def estimate_boiling_point(self, mol):
        try:
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            bp_est = 100 + 0.5 * mw + 0.1 * tpsa + 20 * h_donors + 10 * h_acceptors
            return max(50, min(bp_est, 500))
        except:
            return 0

    def estimate_volatility_index(self, mol):
        try:
            bp_est = self.estimate_boiling_point(mol)
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            if bp_est > 0:
                logVP = 8.0 - 0.01 * bp_est - 0.005 * mw + 0.5 * logp
                vp_pa = 10 ** logVP
                return vp_pa
            return 0
        except:
            return 0

    
    # ---------- 3. Main processing pipeline ----------
    def process_smiles(self, smiles):
        if not smiles or pd.isna(smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Cannot parse SMILES: {smiles}")
                return None

            mol_3d = self.generate_3d_structure(Chem.MolFromSmiles(smiles))
            results = {}

            # 1. Mordred2D
            mordred_2d = self.calculate_mordred_2d(mol)
            if mordred_2d:
                results.update(mordred_2d)

            # 2. Mordred2D,3D
            mordred_2d_3d = self.calculate_mordred_2d_3d(mol_3d)
            if mordred_2d_3d:
                mordred_2d_3d_prefixed = {f'3D_{k}': v for k, v in mordred_2d_3d.items()}
                results.update(mordred_2d_3d_prefixed)

            # 3. ECFP4
            ecfp4 = self.calculate_ecfp4(mol, 1024)
            if ecfp4:
                results.update(ecfp4)

            # 4. Avalon fingerprint
            avalon = self.calculate_avalon_fingerprint(mol, 512)
            if avalon:
                results.update(avalon)

            # 5. MACCS keys
            maccs = self.calculate_maccs_keys(mol)
            if maccs:
                results.update(maccs)

            # 6. RDKit fingerprint
            rdkfp = self.calculate_rdkit_fingerprint(mol, 1024)
            if rdkfp:
                results.update(rdkfp)

            # 7. Domain-specific descriptors
            domain_desc = self.calculate_domain_specific_descriptors(mol)
            if domain_desc:
                results.update(domain_desc)

            return results
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            return None


def calculate_descriptors_for_dataset(input_file, smiles_column='SMILES',
                                      output_file='molecular_descriptors.csv',
                                      n_jobs=1):
    """
    Calculate molecular descriptors for a dataset.

    Parameters
    ----------
    input_file : str
        Path to input file (CSV or Excel)
    smiles_column : str
        Name of the SMILES column
    output_file : str
        Path to output file
    n_jobs : int
        Number of parallel processes (1 for single process)
    """

    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Only CSV or Excel files are supported")

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in file")

    print(f"Found {len(df)} molecules")

    calculator = MolecularDescriptorCalculator()

    all_results = []

    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        print(f"Processing molecule {idx + 1}/{len(df)}: {smiles[:50]}...")

        results = calculator.process_smiles(smiles)

        if results:
            results['SMILES'] = smiles
            for col in df.columns:
                if col != smiles_column:
                    results[col] = row[col]

            all_results.append(results)

    if not all_results:
        print("No descriptors were calculated successfully")
        return None

    result_df = pd.DataFrame(all_results)

    original_cols = [col for col in df.columns if col in result_df.columns]
    desc_cols = [col for col in result_df.columns if col not in original_cols]
    result_df = result_df[original_cols + desc_cols]

    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Descriptors computed for {len(result_df)} molecules")
    print(f"Total features: {len(result_df.columns) - len(original_cols)}")

    mordred_2d_cols = [col for col in result_df.columns if not col.startswith('3D_') and
                       not any(col.startswith(p) for p in ['ECFP4_', 'AFP_', 'MACCS_', 'RDKFP_'])]
    mordred_3d_cols = [col for col in result_df.columns if col.startswith('3D_')]
    ecfp4_cols = [col for col in result_df.columns if col.startswith('ECFP4_')]
    afp_cols = [col for col in result_df.columns if col.startswith('AFP_')]
    maccs_cols = [col for col in result_df.columns if col.startswith('MACCS_')]
    rdkfp_cols = [col for col in result_df.columns if col.startswith('RDKFP_')]

    print("\nDescriptor counts by type:")
    print(f"Mordred2D: {len(mordred_2d_cols)}")
    print(f"Mordred2D,3D: {len(mordred_3d_cols)}")
    print(f"ECFP4: {len(ecfp4_cols)}")
    print(f"Avalon: {len(afp_cols)}")
    print(f"MACCS: {len(maccs_cols)}")
    print(f"RDKFP: {len(rdkfp_cols)}")

    return result_df


def remove_constant_descriptors(df, threshold=0.99):
    """
    Remove constant or near-constant descriptors.
    Handles all-NaN columns without index errors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing descriptors
    threshold : float
        Remove a descriptor if the most common value appears in more than this fraction of rows
    """
    desc_start_idx = 1  # first column is SMILES, descriptors start from index 1

    constant_cols = []

    for col in df.columns[desc_start_idx:]:
        non_null_series = df[col].dropna()

        if len(non_null_series) == 0:
            constant_cols.append(col)
        elif non_null_series.nunique() == 1:
            constant_cols.append(col)
        else:
            value_counts = non_null_series.value_counts(normalize=True)
            if value_counts.iloc[0] > threshold:
                constant_cols.append(col)

    print(f"Removed {len(constant_cols)} constant/near-constant descriptors")

    df_clean = df.drop(columns=constant_cols)

    return df_clean


if __name__ == "__main__":
    # Example usage
    input_file = "your_data.xlsx"      # replace with your file path
    smiles_column = "SMILES"           # replace with your SMILES column name

    result_df = calculate_descriptors_for_dataset(
        input_file=input_file,
        smiles_column=smiles_column,
        output_file="molecular_descriptors_full.csv",
        n_jobs=1
    )

    # Optional: remove constant/near-constant descriptors
    if result_df is not None:
        result_clean = remove_constant_descriptors(result_df)
        result_clean.to_csv("molecular_descriptors_clean.csv", index=False)
        print("Cleaned data saved to: molecular_descriptors_clean.csv")