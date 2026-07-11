#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ELEMENT_NAMES = {
    'H': 'HYDROGEN', 'HE': 'HELIUM', 'LI': 'LITHIUM', 'BE': 'BERYLLIUM', 'B': 'BORON',
    'C': 'CARBON', 'N': 'NITROGEN', 'O': 'OXYGEN', 'F': 'FLUORINE', 'NE': 'NEON',
    'NA': 'SODIUM', 'MG': 'MAGNESIUM', 'AL': 'ALUMINUM', 'SI': 'SILICON', 'P': 'PHOSPHORUS',
    'S': 'SULFUR', 'CL': 'CHLORINE', 'AR': 'ARGON', 'K': 'POTASSIUM', 'CA': 'CALCIUM',
    'SC': 'SCANDIUM', 'TI': 'TITANIUM', 'V': 'VANADIUM', 'CR': 'CHROMIUM', 'MN': 'MANGANESE',
    'FE': 'IRON', 'CO': 'COBALT', 'NI': 'NICKEL', 'CU': 'COPPER', 'ZN': 'ZINC', 'GA': 'GALLIUM',
    'GE': 'GERMANIUM', 'AS': 'ARSENIC', 'SE': 'SELENIUM', 'BR': 'BROMINE', 'KR': 'KRYPTON',
    'RB': 'RUBIDIUM', 'SR': 'STRONTIUM', 'Y': 'YTTRIUM', 'ZR': 'ZIRCONIUM', 'NB': 'NIOBIUM',
    'MO': 'MOLYBDENUM', 'TC': 'TECHNETIUM', 'RU': 'RUTHENIUM', 'RH': 'RHODIUM', 'PD': 'PALLADIUM',
    'AG': 'SILVER', 'CD': 'CADMIUM', 'IN': 'INDIUM', 'SN': 'TIN', 'SB': 'ANTIMONY',
    'TE': 'TELLURIUM', 'I': 'IODINE', 'XE': 'XENON', 'CS': 'CESIUM', 'BA': 'BARIUM',
    'LA': 'LANTHANUM', 'CE': 'CERIUM', 'PR': 'PRASEODYMIUM', 'ND': 'NEODYMIUM', 'PM': 'PROMETHIUM',
    'SM': 'SAMARIUM', 'EU': 'EUROPIUM', 'GD': 'GADOLINIUM', 'TB': 'TERBIUM', 'DY': 'DYSPROSIUM',
    'HO': 'HOLMIUM', 'ER': 'ERBIUM', 'TM': 'THULIUM', 'YB': 'YTTERBIUM', 'LU': 'LUTETIUM',
    'HF': 'HAFNIUM', 'TA': 'TANTALUM', 'W': 'TUNGSTEN', 'RE': 'RHENIUM', 'OS': 'OSMIUM',
    'IR': 'IRIDIUM', 'PT': 'PLATINUM', 'AU': 'GOLD', 'HG': 'MERCURY', 'TL': 'THALLIUM',
    'PB': 'LEAD', 'BI': 'BISMUTH', 'PO': 'POLONIUM', 'AT': 'ASTATINE', 'RN': 'RADON'
}

L_TO_INT = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8}
SYMBOLS_DESC = sorted(ELEMENT_NAMES.keys(), key=len, reverse=True)


@dataclass
class Shell:
    am: str
    exps: List[float]
    coeffs: List[float]


def to_e(x: float) -> str:
    return f"{x:.10E}"


def normalize_element(element: str) -> tuple[str, str]:
    elem = element.strip()
    if not elem:
        raise ValueError('Empty element')
    if len(elem) <= 2 and elem[0].isalpha():
        symbol = elem[0].upper() + elem[1:].lower()
        upper = symbol.upper()
        if upper not in ELEMENT_NAMES:
            raise ValueError(f'Unsupported/unknown element symbol: {element}')
        return symbol, ELEMENT_NAMES[upper]
    upper_name = elem.upper()
    for sym, name in ELEMENT_NAMES.items():
        if name == upper_name:
            return sym.title(), name
    raise ValueError(f'Unsupported/unknown element name: {element}')


def parse_elements_arg(text: str) -> List[str]:
    raw = text.strip()
    if not raw:
        return []
    if ',' in raw or ' ' in raw:
        parts = re.split(r'[\s,]+', raw)
        out = []
        seen = set()
        for p in parts:
            if not p:
                continue
            sym, _ = normalize_element(p)
            if sym not in seen:
                out.append(sym)
                seen.add(sym)
        return out

    out: List[str] = []
    i = 0
    seen = set()
    while i < len(raw):
        matched = False
        for sym in SYMBOLS_DESC:
            if raw[i:i+len(sym)].upper() == sym:
                nice = sym[0] + sym[1:].lower()
                if nice not in seen:
                    out.append(nice)
                    seen.add(nice)
                i += len(sym)
                matched = True
                break
        if not matched:
            raise ValueError(f'Cannot parse element list token near: {raw[i:]}')
    return out


def strip_counter_suffix(name: str) -> str:
    stem = Path(name).name
    stem = re.sub(r'\.[^.]+$', '', stem)
    stem = re.sub(r'\(\d+\)$', '', stem)
    return stem


def canonical_short_label(text: str, role: str, fallback: str) -> str:
    s = text.lower().replace('_', '-').replace(' ', '-')
    s = re.sub(r'-+', '-', s)

    if role == 'ecp':
        m = re.search(r'(ecp\d+[a-z0-9]*)', s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if 'ecp' in s:
            tail = re.sub(r'.*?(ecp)', r'\1', s, flags=re.IGNORECASE)
            return re.sub(r'[^A-Za-z0-9]+', '', tail).upper()
        return fallback

    aug = bool(re.search(r'(^|-)aug(-|$)|(^|-)av?tz|(^|-)avqz|(^|-)avdz|(^|-)av5z', s))

    card_map = [
        ('pvdz', 'vdz'), ('pvtz', 'vtz'), ('pvqz', 'vqz'), ('pv5z', 'v5z'),
        ('vdz', 'vdz'), ('vtz', 'vtz'), ('vqz', 'vqz'), ('v5z', 'v5z'),
    ]
    cardinal = None
    for pat, lab in card_map:
        if pat in s:
            cardinal = lab
            break
    if cardinal is None:
        cardinal = fallback

    prefix = ('a' + cardinal) if aug else cardinal
    pp = '-pp' if '-pp' in s or 'pp-' in s else ''
    f12 = '-f12' if 'f12' in s else ''
    jk = '-jkfit' if 'jkfit' in s else ''
    mp2 = '-mp2fit' if 'mp2fit' in s else ''
    opt = '-optri' if 'optri' in s else ''

    label = prefix + pp + f12 + jk + mp2 + opt
    label = re.sub(r'-+', '-', label).strip('-')
    return label or fallback


def decode_element_group(token: str) -> List[str]:
    token = token.strip()
    if not token or not re.fullmatch(r'[A-Za-z]+', token):
        return []
    try:
        elems = parse_elements_arg(token)
    except ValueError:
        return []
    return elems


def infer_per_element_labels(path: Path, role: str, elements: List[str]) -> Dict[str, str]:
    stem = strip_counter_suffix(path.name)
    tokens = stem.split('_')
    group_map: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        elems = decode_element_group(tokens[i])
        if elems:
            j = i + 1
            desc_tokens: List[str] = []
            while j < len(tokens) and not decode_element_group(tokens[j]):
                desc_tokens.append(tokens[j])
                j += 1
            if desc_tokens:
                desc = '_'.join(desc_tokens)
            elif role == 'ecp' and i > 0:
                desc = '_'.join(tokens[:i])
            else:
                desc = tokens[i]
            for e in elems:
                group_map[e] = desc
            i = j
        else:
            i += 1

    labels: Dict[str, str] = {}
    fallback_map = {
        'basis': 'basis',
        'optri': 'optri',
        'mp2fit': 'mp2fit',
        'jkfit': 'jkfit',
        'ecp': 'ECP',
    }
    for el in elements:
        desc = group_map.get(el, stem)
        labels[el] = canonical_short_label(desc, role, fallback_map[role])
    return labels


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def available_elements_in_basis(path: Path) -> List[str]:
    out = []
    lines = read_text(path).splitlines()
    for line in lines:
        stripped = line.strip().upper()
        for sym, name in ELEMENT_NAMES.items():
            if stripped == name or stripped == sym:
                nice = sym[0] + sym[1:].lower()
                if nice not in out:
                    out.append(nice)
    return out


def extract_basis_block(path: Path, element_symbol: str, element_name_upper: str) -> List[Shell]:
    lines = read_text(path).splitlines()
    target_idx: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper() in {element_name_upper, element_symbol.upper()}:
            target_idx = i + 1
            break
    if target_idx is None:
        raise ValueError(f'Could not find element block for {element_symbol} in {path.name}')

    shells: List[Shell] = []
    i = target_idx
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('$END'):
            break
        if re.fullmatch(r'[A-Z][A-Z ]*', stripped) and stripped.upper() not in L_TO_INT:
            break
        m = re.match(r'^([SPDFGHIJK])\s+(\d+)\s*$', stripped, re.IGNORECASE)
        if not m:
            raise ValueError(f'Unexpected line in {path.name} near line {i+1}: {lines[i]!r}')
        am = m.group(1).upper()
        nprim = int(m.group(2))
        exps: List[float] = []
        coeffs: List[float] = []
        for _ in range(nprim):
            i += 1
            parts = lines[i].split()
            if len(parts) < 3:
                raise ValueError(f'Bad primitive line in {path.name} near line {i+1}: {lines[i]!r}')
            exps.append(float(parts[1].replace('D', 'E').replace('d', 'E')))
            coeffs.append(float(parts[2].replace('D', 'E').replace('d', 'E')))
        shells.append(Shell(am=am, exps=exps, coeffs=coeffs))
        i += 1
    if not shells:
        raise ValueError(f'No shells found for {element_symbol} in {path.name}')
    return shells


def build_coeff_matrix(shells: List[Shell]) -> tuple[List[float], List[List[float]]]:
    """Convert repeated ORCA/GAMESS-style shells into one GENBAS block.

    GENBAS wants, for each angular momentum, one primitive pool plus a coefficient
    matrix of shape (n_primitive, n_contraction). If several shells reuse the same
    exponent set, the exponents must appear only once in the primitive pool.
    """
    primitive_index: Dict[str, int] = {}
    primitive_values: List[float] = []
    matrix: List[List[float]] = []

    def exp_key(x: float) -> str:
        return f"{x:.12g}"

    ncontr = len(shells)
    for col, sh in enumerate(shells):
        for exp, coeff in zip(sh.exps, sh.coeffs):
            key = exp_key(exp)
            if key not in primitive_index:
                primitive_index[key] = len(primitive_values)
                primitive_values.append(exp)
                matrix.append([0.0] * ncontr)
            row = primitive_index[key]
            matrix[row][col] = coeff
    return primitive_values, matrix


def basis_shells_to_genbas(symbol: str, label: str, title: str, shells: List[Shell]) -> str:
    grouped: dict[str, List[Shell]] = defaultdict(list)
    order: List[str] = []
    for shell in shells:
        if shell.am not in grouped:
            order.append(shell.am)
        grouped[shell.am].append(shell)

    am_ints = [str(L_TO_INT[am]) for am in order]
    ncontr = [str(len(grouped[am])) for am in order]
    primitive_blocks = [build_coeff_matrix(grouped[am])[0] for am in order]
    nprim = [str(len(prims)) for prims in primitive_blocks]

    out: List[str] = []
    out.append(f'{symbol.upper()}:{label}')
    out.append(title)
    out.append('')
    out.append(str(len(order)))
    out.append(' '.join(am_ints))
    out.append(' '.join(ncontr))
    out.append(' '.join(nprim))
    out.append('')

    for am in order:
        prims, coeff_matrix = build_coeff_matrix(grouped[am])
        for start in range(0, len(prims), 5):
            out.append(' '.join(to_e(x) for x in prims[start:start+5]))
        out.append('')
        for row in coeff_matrix:
            for start in range(0, len(row), 7):
                out.append(' '.join(to_e(x) for x in row[start:start+7]))
        out.append('')

    return '\n'.join(out).rstrip() + '\n'


def parse_ecp(path: Path, element_symbol: str) -> tuple[str, int, str, list[tuple[str, list[tuple[float, int, float]]]]]:
    lines = [ln.strip() for ln in read_text(path).splitlines() if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf'^{re.escape(element_symbol)}\s+0$', ln, re.IGNORECASE):
            start = i
            break
    if start is None:
        raise ValueError(f'Could not find ECP section for {element_symbol} in {path.name}')

    header = lines[start + 1].split()
    if len(header) < 3:
        raise ValueError(f'Bad ECP header in {path.name}: {lines[start + 1]!r}')
    ecp_name = header[0]
    ncore = int(header[2])

    channels: list[tuple[str, list[tuple[float, int, float]]]] = []
    i = start + 2
    local_l = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('!'):
            i += 1
            continue
        if re.match(r'^[A-Z][a-z]?\s+0$', ln):
            break
        if ln.upper().endswith('-KOMPONENTE') or ln.upper().endswith('KOMPONENTE'):
            m = re.match(r'^([SPDFGHIJK])\-?Komponente$', ln, re.IGNORECASE)
            if not m:
                raise ValueError(f'Cannot parse local ECP component line: {ln!r}')
            local_l = m.group(1).lower()
            ch_name = local_l
        else:
            m = re.match(r'^([SPDFGHIJK])\-[SPDFGHIJK]$', ln, re.IGNORECASE)
            if not m:
                i += 1
                continue
            ch_name = m.group(1).lower()
        i += 1
        nterms = int(lines[i].split()[0])
        terms: list[tuple[float, int, float]] = []
        for _ in range(nterms):
            i += 1
            parts = lines[i].split()
            if len(parts) < 3:
                raise ValueError(f'Bad ECP term in {path.name}: {lines[i]!r}')
            power = int(float(parts[0]))
            exponent = float(parts[1].replace('D', 'E').replace('d', 'E'))
            coef = float(parts[2].replace('D', 'E').replace('d', 'E'))
            terms.append((coef, power, exponent))
        channels.append((ch_name, terms))
        i += 1
    if local_l is None:
        raise ValueError(f'Could not identify local angular momentum in {path.name}')
    return ecp_name, ncore, local_l, channels


def ecp_to_genbas(symbol: str, label: str, title: str, parsed_ecp: tuple[str, int, str, list[tuple[str, list[tuple[float, int, float]]]]]) -> str:
    _ecp_name, ncore, local_l, channels = parsed_ecp
    lmax = L_TO_INT[local_l.upper()]
    out: List[str] = []
    out.append('*')
    out.append(f'{symbol.upper()}:{label}')
    out.append(title)
    out.append('*')
    out.append(f'    NCORE = {ncore}    LMAX = {lmax}')
    for ch_name, terms in channels:
        if ch_name == local_l:
            out.append(local_l)
        else:
            out.append(f'{ch_name}-{local_l}')
        for coef, power, exponent in terms:
            out.append(f'{to_e(coef)} {power:d} {to_e(exponent)}')
    out.append('*')
    return '\n'.join(out).rstrip() + '\n'


def generate_genbas(
    output_path: Path,
    elements: List[str],
    basis_file: Optional[Path] = None,
    optri_file: Optional[Path] = None,
    jkfit_file: Optional[Path] = None,
    mp2fit_file: Optional[Path] = None,
    ecp_file: Optional[Path] = None,
) -> dict[str, Optional[Path]]:
    found: dict[str, Optional[Path]] = {
        'basis': basis_file,
        'optri': optri_file,
        'mp2fit': mp2fit_file,
        'jkfit': jkfit_file,
        'ecp': ecp_file,
    }

    if all(p is None for p in found.values()):
        raise ValueError('No input files provided. Pass at least one of basis/optri/jkfit/mp2fit/ecp.')

    for role, p in found.items():
        if p is not None and not p.is_file():
            raise ValueError(f'{role} file not found: {p}')

    for role in ['basis', 'optri', 'mp2fit', 'jkfit']:
        p = found[role]
        if p is None:
            continue
        available = set(available_elements_in_basis(p))
        unavailable = [el for el in elements if el not in available]
        if unavailable:
            raise ValueError(f'{role} file is missing element block(s): ' + ', '.join(unavailable))

    basis_labels = infer_per_element_labels(found['basis'], 'basis', elements) if found['basis'] else {}
    optri_labels = infer_per_element_labels(found['optri'], 'optri', elements) if found['optri'] else {}
    mp2fit_labels = infer_per_element_labels(found['mp2fit'], 'mp2fit', elements) if found['mp2fit'] else {}
    jkfit_labels = infer_per_element_labels(found['jkfit'], 'jkfit', elements) if found['jkfit'] else {}
    ecp_labels = infer_per_element_labels(found['ecp'], 'ecp', elements) if found['ecp'] else {}

    parts: List[str] = []
    for el in elements:
        sym, name_upper = normalize_element(el)
        if found['basis'] is not None:
            parts.append(basis_shells_to_genbas(sym, basis_labels[el], f'{sym} {basis_labels[el]} converted from {found["basis"].name}', extract_basis_block(found['basis'], sym, name_upper)))
        if found['jkfit'] is not None:
            parts.append(basis_shells_to_genbas(sym, jkfit_labels[el], f'{sym} {jkfit_labels[el]} converted from {found["jkfit"].name}', extract_basis_block(found['jkfit'], sym, name_upper)))
        if found['mp2fit'] is not None:
            parts.append(basis_shells_to_genbas(sym, mp2fit_labels[el], f'{sym} {mp2fit_labels[el]} converted from {found["mp2fit"].name}', extract_basis_block(found['mp2fit'], sym, name_upper)))
        if found['optri'] is not None:
            parts.append(basis_shells_to_genbas(sym, optri_labels[el], f'{sym} {optri_labels[el]} converted from {found["optri"].name}', extract_basis_block(found['optri'], sym, name_upper)))
        if found['ecp'] is not None:
            try:
                parsed = parse_ecp(found['ecp'], sym)
            except Exception:
                parsed = None
            if parsed is not None:
                ecp_label = ecp_labels.get(el, 'ECP')
                parts.append(ecp_to_genbas(sym, ecp_label, f'{sym} {ecp_label} converted from {found["ecp"].name}', parsed))

    output_path.write_text('\n'.join(parts), encoding='utf-8')
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate MRCC GENBAS for multiple elements from AO / OptRI / MP2Fit / JKFit / ECP files.')
    parser.add_argument('-e', '--elements', default='H,C,N,O,Zn', help='Elements to include, e.g. H,C,N,O,Zn or HCNOZn')
    parser.add_argument('-o', '--output', type=Path, default=Path('GENBAS'), help='Output GENBAS file path')
    parser.add_argument('--basis-file', type=Path, default=None, help='Explicit AO basis file (default: None)')
    parser.add_argument('--optri-file', type=Path, default=None, help='Explicit OptRI file (default: None)')
    parser.add_argument('--mp2fit-file', type=Path, default=None, help='Explicit MP2Fit file (default: None)')
    parser.add_argument('--jkfit-file', '--jkft-file', dest='jkfit_file', type=Path, default=None, help='Explicit JKFit file (default: None)')
    parser.add_argument('--ecp-file', type=Path, default=None, help='Explicit ECP file (default: None)')
    args = parser.parse_args()

    try:
        elements = parse_elements_arg(args.elements)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    if not elements:
        print('Error: no elements requested.', file=sys.stderr)
        return 2

    try:
        found = generate_genbas(
            output_path=args.output,
            elements=elements,
            basis_file=args.basis_file,
            optri_file=args.optri_file,
            jkfit_file=args.jkfit_file,
            mp2fit_file=args.mp2fit_file,
            ecp_file=args.ecp_file,
        )
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2

    print(f'Wrote {args.output}')
    print('Detected files:')
    for role in ['basis', 'optri', 'mp2fit', 'jkfit', 'ecp']:
        print(f'  {role:7s}: {found[role]}')
    print('Element-wise labels:')
    for el in elements:
        fields: List[str] = []
        if found['basis'] is not None:
            fields.append(f'basis={infer_per_element_labels(found["basis"], "basis", elements)[el]}')
        if found['jkfit'] is not None:
            fields.append(f'dfbasis_scf={infer_per_element_labels(found["jkfit"], "jkfit", elements)[el]}')
        if found['mp2fit'] is not None:
            fields.append(f'dfbasis_cor={infer_per_element_labels(found["mp2fit"], "mp2fit", elements)[el]}')
        if found['optri'] is not None:
            fields.append(f'dfbasis_cab={infer_per_element_labels(found["optri"], "optri", elements)[el]}')
        print(f'  [{el}] ' + ' '.join(fields), end='')
        if found['ecp'] is not None and el in infer_per_element_labels(found['ecp'], 'ecp', elements):
            try:
                parse_ecp(found['ecp'], el)
                print(f' ecp={infer_per_element_labels(found["ecp"], "ecp", elements)[el]}')
            except Exception:
                print()
        else:
            print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
