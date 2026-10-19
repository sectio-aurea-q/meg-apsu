// MEG-APSU — structure-based triage of hydrogen-tunnelling enzymes
// sectio-aurea-q · MEGALODON Research · 2026
//
// Question: does the active site in this PDB structure contain a catalytically
// competent centre for hydrogen transfer (radical C-H abstraction, hydride transfer)?
//
// Core (sectio-aurea-q): motif rules on the structure (classify_sites) plus a physical
// filter that separates catalytic metal/heme centres from electron-transfer and
// transport sites (metal_catalytic, v1.3). Fast, explainable, no training.
//
// v1.5 (19.10.2026): labels and commands only — every score is bit-identical to v1.4.
// v1.2: Nucleotide-gated kinase exclusion, SAM methyltransferase, 3 PDB fixes
use ndarray::Array2;
use num_complex::Complex64;
use num_traits::Zero;
use serde::Serialize;
use std::{env, fs};
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering as AO};
mod bench;

// v1.3: Katalyse-Filter für Metallzentren. false = altes v1.2-Verhalten (für Vergleiche).
pub static V13: AtomicBool = AtomicBool::new(true);

const HBAR: f64 = 1.054571817e-34;
const KB: f64 = 1.380649e-23;
const MP: f64 = 1.67262192369e-27;
const EV_J: f64 = 1.602176634e-19;
const ANG_M: f64 = 1e-10;
type Rho = Array2<Complex64>;
fn cr(r: f64) -> Complex64 { Complex64::new(r, 0.0) }
fn ci(i: f64) -> Complex64 { Complex64::new(0.0, i) }

// ═══ LINDBLAD RK4 ═══════════════════════════════════════════════════════
fn dagger(m: &Rho) -> Rho { m.t().mapv(|z| z.conj()) }
fn comm(a: &Rho, b: &Rho) -> Rho { a.dot(b) - b.dot(a) }
fn anticomm(a: &Rho, b: &Rho) -> Rho { a.dot(b) + b.dot(a) }
struct Noise { l: Rho, ldl: Rho, g: f64 }
impl Noise {
    fn new(l: Rho, g: f64) -> Self { let ld = dagger(&l); Noise { l: l.clone(), ldl: ld.dot(&l), g } }
    fn diss(&self, rho: &Rho) -> Rho {
        let ld = dagger(&self.l);
        (self.l.dot(rho).dot(&ld) - anticomm(&self.ldl, rho) * cr(0.5)) * cr(self.g)
    }
}
struct Sys { h: Rho, ops: Vec<Noise>, mi: Complex64 }
impl Sys {
    fn new(h: Rho) -> Self { Sys { h, ops: Vec::new(), mi: ci(-1.0 / HBAR) } }
    fn add(&mut self, l: Rho, g: f64) { self.ops.push(Noise::new(l, g)); }
    fn deriv(&self, rho: &Rho) -> Rho {
        let mut d = comm(&self.h, rho) * self.mi;
        for op in &self.ops { d = d + op.diss(rho); }
        d
    }
}
fn rk4(s: &Sys, r: &Rho, dt: f64) -> Rho {
    let h = cr(dt * 0.5);
    let k1 = s.deriv(r); let k2 = s.deriv(&(r + &(&k1 * h)));
    let k3 = s.deriv(&(r + &(&k2 * h))); let k4 = s.deriv(&(r + &(&k3 * cr(dt))));
    r + &((&k1 + &(&k2 * cr(2.0)) + &(&k3 * cr(2.0)) + &k4) * cr(dt / 6.0))
}
fn noise_2site(sys: &mut Sys, temp: f64) {
    let n = 2; let kt = KB * temp;
    let gd = kt / (HBAR * 50.0);
    for i in 0..n { let mut l = Rho::zeros((n,n)); l[[i,i]] = cr(1.0); sys.add(l, gd); }
    let gr = 1.0 / 500.0e-15;
    let de = (sys.h[[1,1]].re - sys.h[[0,0]].re).abs();
    let nb = if de > 1e-30 { 1.0/((de/kt).exp()-1.0).max(0.01) } else { kt/1e-22 };
    let mut ld = Rho::zeros((n,n)); ld[[0,1]] = cr(1.0); sys.add(ld, gr*(nb+1.0)*0.1);
    let mut lu = Rho::zeros((n,n)); lu[[1,0]] = cr(1.0); sys.add(lu, gr*nb*0.1);
    let mut lv = Rho::zeros((n,n)); lv[[0,0]] = cr(1.0); lv[[1,1]] = cr(-1.0);
    sys.add(lv, 1.0/200.0e-15*0.02);
    for i in 0..n { let mut ls = Rho::zeros((n,n)); ls[[i,i]] = cr(1.0); sys.add(ls, 1.0/1.0e-12*0.005); }
}
/// Class prior from a two-level Lindblad model.
/// NOTE (v1.5): barrier and distance are FIXED per motif type (see RxnType::barrier_ev and
/// the class distances in compute_qvs), not read from atomic coordinates. The result is
/// therefore a constant per motif type and enters the score as a class prior. A version
/// that takes donor-acceptor geometry from the structure is planned (v2.0).
fn lindblad_enh(barrier_ev: f64, dist_ang: f64, temp: f64) -> f64 {
    let n = 2; let omega0 = 1e13;
    let tw = (dist_ang - 1.0).max(0.2) * ANG_M;
    let kappa = (2.0 * MP * barrier_ev * EV_J).sqrt() / HBAR;
    let wkb_t = (-2.0 * kappa * tw).exp();
    let epsilon = barrier_ev * 0.1 * EV_J;
    let delta = HBAR * omega0 * wkb_t.sqrt();
    let mut h = Rho::zeros((n,n));
    h[[0,0]] = cr(0.0); h[[1,1]] = cr(epsilon); h[[0,1]] = cr(delta); h[[1,0]] = cr(delta);
    let mut sys = Sys::new(h); noise_2site(&mut sys, temp);
    let mut rho = Rho::zeros((n,n)); rho[[0,0]] = cr(1.0);
    let dt = 1e-15; let steps = 2000; let mut max_p1 = 0.0_f64;
    for step in 0..steps {
        rho = rk4(&sys, &rho, dt);
        if step % 100 == 0 {
            for i in 0..n { if rho[[i,i]].re < 0.0 { rho[[i,i]] = Complex64::zero(); } }
            let tr: f64 = (0..n).map(|i| rho[[i,i]].re).sum();
            if tr > 1e-15 { rho.mapv_inplace(|z| z * cr(1.0/tr)); }
        }
        let tr: f64 = (0..n).map(|i| rho[[i,i]].re).sum();
        if tr.is_nan() || (tr-1.0).abs() > 0.2 { break; }
        if rho[[1,1]].re > max_p1 { max_p1 = rho[[1,1]].re; }
    }
    let q_rate = if max_p1 > 0.001 { max_p1 * omega0 } else { omega0 * wkb_t };
    let c_rate = omega0 * (-barrier_ev * EV_J / (KB * temp)).exp();
    if c_rate > 1e-100 { q_rate / c_rate } else { 1e50 }
}

// ═══ PDB PARSER ═════════════════════════════════════════════════════════
#[derive(Clone, Debug)]
struct Atom { name: String, res: String, chain: char, seq: i32,
    x: f64, y: f64, z: f64, elem: String, is_het: bool }
fn parse_pdb(c: &str) -> (String, Vec<Atom>) {
    let mut title = String::new(); let mut atoms = Vec::new();
    for line in c.lines() {
        if line.starts_with("TITLE") {
            let t = line.get(10..line.len().min(80)).unwrap_or("").trim();
            if !t.is_empty() { if !title.is_empty() { title.push(' '); } title.push_str(t); }
        }
        let is_het = line.starts_with("HETATM");
        if !line.starts_with("ATOM") && !is_het { continue; }
        if line.len() < 54 { continue; }
        let alt = line.as_bytes().get(16).copied().unwrap_or(b' ');
        if alt != b' ' && alt != b'A' { continue; }
        let name = line.get(12..16).unwrap_or("").trim().to_string();
        let res = line.get(17..20).unwrap_or("").trim().to_string();
        let chain = line.get(21..22).unwrap_or(" ").chars().next().unwrap_or(' ');
        let seq: i32 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);
        let x: f64 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f64 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f64 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let elem = if line.len() >= 78 { line.get(76..78).unwrap_or("").trim().to_string() }
                   else { name.chars().next().unwrap_or('C').to_string() };
        if elem == "H" || elem == "D" { continue; }
        if res == "HOH" || res == "WAT" { continue; }
        atoms.push(Atom { name, res, chain, seq, x, y, z, elem, is_het });
    }
    if title.is_empty() { title = "Unknown".into(); }
    (title, atoms)
}
fn adist(a: &Atom, b: &Atom) -> f64 {
    ((a.x-b.x).powi(2)+(a.y-b.y).powi(2)+(a.z-b.z).powi(2)).sqrt()
}

// ═══ COFACTOR DATABASE (v1.1: CLEANED) ══════════════════════════════════
// v1.1 FIX: Removed PTR, STY, CRO, TYS, OMY, NIY — these are modified
// amino acids, NOT quinone cofactors. This was causing false positives.
fn is_heme(r: &str) -> bool {
    // v1.2: Added COH (cobalt-protoporphyrin, used in COX-2 crystal structures)
    matches!(r, "HEM"|"HEC"|"HEA"|"HEB"|"HEG"|"HEO"|"1HE"|"2HE"|"DHE"
        |"PP9"|"PPH"|"HAS"|"SRM"|"MHM"|"HDD"|"HME"|"HNI"|"FDE"|"DDH"|"MMP"
        |"COH")
}
fn is_quinone(r: &str) -> bool {
    // v1.1: ONLY actual quinone cofactors. NO modified amino acids.
    matches!(r, "TPQ"|"TTQ"|"CTQ"|"LTQ"|"TRQ")
}
fn is_folate(r: &str) -> bool {
    // v1.2.1: Removed FMT (formate ion, crystallization buffer — NOT a folate cofactor)
    matches!(r, "THF"|"DHF"|"FOL"|"MTX"|"UMP"|"DUM"|"TMP"|"5FU"
        |"W12"|"U5P"|"CH2"|"M5F"|"5FC"|"H4B"|"BH4"|"BH2"
        |"THG"|"MTE"|"10F"|"LMT"|"AMT"|"THP")
}
fn is_nad_fad(r: &str) -> bool {
    matches!(r, "NAD"|"NAP"|"FAD"|"FMN"|"NAI"|"NDP"|"NMN"|"FDA"|"FAO")
}
fn is_2og(r: &str) -> bool {
    matches!(r, "AKG"|"2OG"|"OGA"|"KGR"|"NOG"|"MLI"|"SUC")
}
fn is_pterin(r: &str) -> bool {
    matches!(r, "H4B"|"BH4"|"BH2"|"HBI"|"PTE"|"BPH")
}
fn is_sam(r: &str) -> bool {
    // v1.2.1: Removed MSE (selenomethionine, crystallographic phasing agent — NOT SAM)
    matches!(r, "SAM"|"SAH"|"DNC"|"5MC"|"SMM")
}
fn is_nucleotide_substrate(r: &str) -> bool {
    // v1.2: Nucleotides AND metabolic substrates near metals = kinase/transferase, NOT radical
    matches!(r, "ATP"|"ADP"|"AMP"|"GTP"|"GDP"|"ANP"|"ACP"|"AGS"|"ADN"|"APC"|"AP5"|"SEP"|"TPO"
        |"PYR"|"OAA"|"PEP"|"G6P"|"F6P"|"FBP"|"GAP"|"CTP"|"UTP"|"UDP"|"CDP")
}

// ═══ v1.3: IST DIESES METALLZENTRUM KATALYTISCH? ═════════════════════════
// Physikalische Regel, unabhängig von Enzymnamen: Eine C-H-Spaltung am Metall braucht
// eine freie (oder durch Substrat/O2 besetzbare) Koordinationsstelle. Gesättigte Zentren
// sind Elektronentransfer- oder Transportstellen.
//  (a) Fe-S-Cluster / Rubredoxin: ≥4 Schwefel-Liganden, tetraedrisch gesättigt
//  (b) Typ-1-Cu (Cupredoxin): Cu mit Cys-Thiolat-Ligand → Elektronentransfer
//  (c) Häm mit zwei axialen Protein-Liganden (His/His, His/Met) → Cytochrom (ET)
//  (d) Häm mit nur proximalem His: katalytisch nur mit "Push-Pull" (Poulos/Kraut):
//      distaler Säure/Base-Partner (Arg/Gln/Asn) oder gebundener Ligand im distalen Raum.
//      Sonst O2-Transport (Globine).
//  (e) Nicht-Häm-Metall mit Koordinationszahl ≥6 ohne Wasser → gesättigt (z. B. Transferrin)
const LIG_CUT: f64 = 2.6;
fn is_metal(e: &str) -> bool { matches!(e, "FE"|"CU"|"MN"|"CO"|"MO"|"ZN"|"NI"|"MG"|"CA"|"NA"|"K") }
// Austauschbare Kleinliganden: Wasser/Hydroxid/Oxo und zweiatomige Gase bzw. Anionen.
// Sie besetzen die offene Stelle nur vorübergehend → Stelle gilt als offen, und sie sind kein Substrat.
fn is_exchangeable(r: &str) -> bool { matches!(r, "HOH"|"WAT"|"OH"|"O"|"OXO"|"PER"|"OXY"|"O2"|"CMO"|"NO"|"CYN"|"AZI"|"F"|"CL") }
fn metal_catalytic(atoms: &[Atom], mi: usize) -> bool {
    let m = &atoms[mi];
    let lig: Vec<usize> = atoms.iter().enumerate()
        .filter(|(j,a)| *j != mi && !is_metal(&a.elem) && !(a.is_het && is_exchangeable(&a.res)) && adist(m, a) < LIG_CUT)
        .map(|(j,_)| j).collect();
    if lig.len() < 2 { return true; } // schlecht aufgelöst → keine Aussage, altes Verhalten
    let n_s = lig.iter().filter(|&&j| atoms[j].elem == "S").count();
    if lig.len() >= 4 && n_s >= 4 { return false; }                                   // (a)
    if m.elem == "CU" && lig.iter().any(|&j| atoms[j].res=="CYS" && atoms[j].name=="SG") { return false; } // (b)
    let heme_n = lig.iter().filter(|&&j| is_heme(&atoms[j].res) && atoms[j].elem=="N").count();
    if heme_n >= 3 {
        let axial: Vec<usize> = lig.iter().copied().filter(|&j| !atoms[j].is_het).collect();
        if axial.len() >= 2 { return false; }                                         // (c)
        if axial.is_empty() { return true; }
        let ax = &atoms[axial[0]];
        if (ax.res=="CYS" && ax.name=="SG") || (ax.res=="TYR" && ax.name=="OH") { return true; } // Thiolat/Phenolat-Push
        // Normale vom proximalen Liganden durch das Fe auf die distale Seite
        let (nx,ny,nz) = (m.x-ax.x, m.y-ax.y, m.z-ax.z);
        let nl = (nx*nx+ny*ny+nz*nz).sqrt().max(1e-6);
        let distal = |a: &Atom| -> f64 { ((a.x-m.x)*nx + (a.y-m.y)*ny + (a.z-m.z)*nz) / nl };
        let pull = atoms.iter().any(|a| !a.is_het && adist(m,a) < 6.0 && distal(a) > 1.5
            && matches!((a.res.as_str(), a.name.as_str()),
                ("ARG","NE")|("ARG","NH1")|("ARG","NH2")|("GLN","NE2")|("GLN","OE1")|("ASN","ND2")|("ASN","OD1")));
        let ligand = atoms.iter().any(|a| a.is_het && !is_heme(&a.res) && !is_metal(&a.elem) && !is_exchangeable(&a.res)
            && !matches!(a.res.as_str(), "SO4"|"PO4"|"GOL"|"EDO"|"ACT"|"CL"|"IOD"|"BR"|"NO3"|"SCN"|"FMT"|"NH4"|"MPD"|"DMS"|"PEG"|"PGE"|"BME")
            && adist(m,a) < 6.0 && distal(a) > 1.0);
        return pull || ligand;                                                         // (d)
    }
    if lig.len() >= 6 { return false; }                                               // (e)
    true
}

// ═══ REACTION TYPES ═════════════════════════════════════════════════════
#[derive(Clone, Debug, Serialize)]
enum RxnType { RadicalCH, HydrideTransfer, ProtonRelay, LewisAcid }
impl RxnType {
    fn barrier_ev(&self) -> f64 { match self {
        RxnType::RadicalCH => 0.65, RxnType::HydrideTransfer => 0.50,
        RxnType::ProtonRelay => 0.18, RxnType::LewisAcid => 0.30,
    }}
    fn label(&self) -> &str { match self {
        RxnType::RadicalCH => "Radical C-H", RxnType::HydrideTransfer => "Hydride Transfer",
        RxnType::ProtonRelay => "Proton Relay", RxnType::LewisAcid => "Lewis Acid",
    }}
}
#[derive(Clone, Debug, Serialize)]
struct ActiveSite { kind: String, rxn: String, barrier: f64, residues: Vec<String>,
    has_substrate: bool, substrate_name: String }

// ═══ SITE CLASSIFICATION ════════════════════════════════════════════════
fn classify_sites(atoms: &[Atom]) -> Vec<ActiveSite> {
    let mut sites = Vec::new();
    let fe: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="FE").map(|(i,_)|i).collect();
    let cu: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="CU").map(|(i,_)|i).collect();
    let mn: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="MN").map(|(i,_)|i).collect();
    let zn: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="ZN").map(|(i,_)|i).collect();
    let co: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="CO").map(|(i,_)|i).collect();
    let mo: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.elem=="MO").map(|(i,_)|i).collect();
    let his: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="HIS"&&(a.name=="NE2"||a.name=="ND1")).map(|(i,_)|i).collect();
    let ser: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="SER"&&a.name=="OG").map(|(i,_)|i).collect();
    let cys: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="CYS"&&a.name=="SG").map(|(i,_)|i).collect();
    let asp: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="ASP"&&(a.name=="OD1"||a.name=="OD2")).map(|(i,_)|i).collect();
    let glu: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="GLU"&&(a.name=="OE1"||a.name=="OE2")).map(|(i,_)|i).collect();
    let tyr: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| a.res=="TYR"&&a.name=="OH").map(|(i,_)|i).collect();
    let het: Vec<usize> = atoms.iter().enumerate()
        .filter(|(_,a)| a.is_het && !matches!(a.res.as_str(),"HOH"|"WAT"|"SO4"|"PO4"|"GOL"|"EDO"|"ACT"|"CL"|"NA"|"K"|"MG"|"CA"|"ZN"|"IOD"|"BR"|"NO3"|"SCN"|"FMT"|"NH4"|"MPD"|"DMS"|"PEG"|"PGE"|"BME"))
        .map(|(i,_)|i).collect();
    let heme_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_heme(&a.res)).map(|(i,_)|i).collect();
    let quin_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_quinone(&a.res)).map(|(i,_)|i).collect();
    let fol_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_folate(&a.res)).map(|(i,_)|i).collect();
    let nad_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_nad_fad(&a.res)).map(|(i,_)|i).collect();
    let og_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_2og(&a.res)).map(|(i,_)|i).collect();
    let pte_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_pterin(&a.res)).map(|(i,_)|i).collect();
    let sam_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_sam(&a.res)).map(|(i,_)|i).collect();
    let nuc_at: Vec<usize> = atoms.iter().enumerate().filter(|(_,a)| is_nucleotide_substrate(&a.res)).map(|(i,_)|i).collect();

    let v13 = V13.load(AO::SeqCst);
    let mut cat = vec![true; atoms.len()];
    if v13 { for &mi in fe.iter().chain(cu.iter()).chain(mn.iter()).chain(co.iter()).chain(mo.iter()) {
        cat[mi] = metal_catalytic(atoms, mi);
    }}
    let find_sub = |ci: usize, r: f64| -> (bool, String) {
        for &hi in &het { if adist(&atoms[ci], &atoms[hi]) < r { return (true, atoms[hi].res.clone()); }}
        (false, String::new())
    };
    let coord_atoms = |mi: usize, rad: f64| -> Vec<String> {
        atoms.iter().enumerate().filter(|(idx, a)| *idx != mi && !a.is_het && adist(&atoms[mi], a) < rad
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|(_, a)| format!("{}{}:{}", a.res, a.seq, a.name)).collect()
    };

    // 1. Metal-oxo / Metal-heme → RadicalCH
    // v1.2: GATE — if nucleotide (ATP/GTP) is within 6Å of metal, this is a kinase/ATPase, skip
    for &mi in fe.iter().chain(cu.iter()).chain(mn.iter()).chain(co.iter()).chain(mo.iter()) {
        let nuc_nearby = nuc_at.iter().any(|&ni| adist(&atoms[mi], &atoms[ni]) < 6.0);
        if nuc_nearby { continue; } // kinase, not radical enzyme
        if v13 && !cat[mi] { continue; } // v1.3: gesättigtes ET-/Transportzentrum
        let in_heme = heme_at.iter().any(|&hi| adist(&atoms[mi], &atoms[hi]) < 4.5);
        let rad = if in_heme { 4.5 } else { 3.5 };
        let coord = coord_atoms(mi, rad);
        let min_c = if in_heme { 1 } else { 2 };
        if coord.len() >= min_c {
            let (hs, sn) = find_sub(mi, 6.0);
            let k = if in_heme { format!("{}-heme", atoms[mi].elem) } else { format!("{}-oxo", atoms[mi].elem) };
            sites.push(ActiveSite { kind: k, rxn: RxnType::RadicalCH.label().into(),
                barrier: RxnType::RadicalCH.barrier_ev(), residues: coord,
                has_substrate: hs || in_heme,
                substrate_name: if hs { sn } else if in_heme { "HEM".into() } else { String::new() }});
        }
    }

    // 2. NAD/FAD → Hydride
    if !nad_at.is_empty() {
        let c = &atoms[nad_at[0]];
        let nr: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(c, a) < 5.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        sites.push(ActiveSite { kind: format!("NAD/FAD ({})", atoms[nad_at[0]].res),
            rxn: RxnType::HydrideTransfer.label().into(), barrier: RxnType::HydrideTransfer.barrier_ev(),
            residues: nr, has_substrate: true,
            substrate_name: atoms[nad_at[0]].res.clone() });
    }

    // 3. Quinone (TPQ/TTQ/CTQ ONLY) → RadicalCH
    if !quin_at.is_empty() {
        let c = &atoms[quin_at[0]];
        let nr: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(c, a) < 5.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        sites.push(ActiveSite { kind: format!("Quinone ({})", atoms[quin_at[0]].res),
            rxn: RxnType::RadicalCH.label().into(), barrier: RxnType::RadicalCH.barrier_ev() * 0.95,
            residues: nr, has_substrate: true, substrate_name: atoms[quin_at[0]].res.clone() });
    }
    // Cu + Tyr/Trp fallback for amine oxidases without resolved quinone
    if quin_at.is_empty() {
        for &ci in &cu {
            let has_aro = atoms.iter().any(|a| !a.is_het && adist(&atoms[ci], a) < 6.0
                && ((a.res=="TYR" && a.name=="OH") || (a.res=="TRP" && a.name=="NE1")));
            if has_aro && !sites.iter().any(|s| s.kind.contains("CU")||s.kind.contains("Cu")) {
                let nr = coord_atoms(ci, 5.0);
                sites.push(ActiveSite { kind: "Cu-amine oxidase".into(),
                    rxn: RxnType::RadicalCH.label().into(), barrier: RxnType::RadicalCH.barrier_ev()*0.95,
                    residues: nr, has_substrate: true, substrate_name: "Cu-TPQ".into() });
            }
        }
    }

    // 4. Folate → Hydride
    if !fol_at.is_empty() {
        let c = &atoms[fol_at[0]];
        let nr: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(c, a) < 5.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        sites.push(ActiveSite { kind: format!("Folate ({})", atoms[fol_at[0]].res),
            rxn: RxnType::HydrideTransfer.label().into(), barrier: RxnType::HydrideTransfer.barrier_ev(),
            residues: nr, has_substrate: true, substrate_name: atoms[fol_at[0]].res.clone() });
    }

    // 5. Pterin (BH4) → RadicalCH (aromatic amino acid hydroxylases)
    if !pte_at.is_empty() {
        let c = &atoms[pte_at[0]];
        let nr: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(c, a) < 5.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        sites.push(ActiveSite { kind: format!("Pterin ({})", atoms[pte_at[0]].res),
            rxn: RxnType::RadicalCH.label().into(), barrier: RxnType::RadicalCH.barrier_ev() * 0.85,
            residues: nr, has_substrate: true, substrate_name: atoms[pte_at[0]].res.clone() });
    }

    // 5b. v1.2: SAM (S-adenosylmethionine) → Hydride/Methyl Transfer
    // COMT and other methyltransferases use SAM for methyl group transfer with tunneling
    if !sam_at.is_empty() {
        let c = &atoms[sam_at[0]];
        let nr: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(c, a) < 5.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"|"OH"|"NZ"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        sites.push(ActiveSite { kind: format!("SAM methyltransfer ({})", atoms[sam_at[0]].res),
            rxn: RxnType::HydrideTransfer.label().into(), barrier: RxnType::HydrideTransfer.barrier_ev() * 0.9,
            residues: nr, has_substrate: true, substrate_name: atoms[sam_at[0]].res.clone() });
    }

    // 6. Ser-His-Asp / Cys-His-Asp → ProtonRelay
    for &si in &ser { for &hi in &his {
        if adist(&atoms[si], &atoms[hi]) > 4.0 { continue; }
        for &di in &asp {
            if adist(&atoms[hi], &atoms[di]) > 4.0 { continue; }
            let (hs,sn) = find_sub(hi, 6.0);
            sites.push(ActiveSite { kind: "Ser-His-Asp".into(), rxn: RxnType::ProtonRelay.label().into(),
                barrier: RxnType::ProtonRelay.barrier_ev(),
                residues: vec![format!("SER{}",atoms[si].seq),format!("HIS{}",atoms[hi].seq),format!("ASP{}",atoms[di].seq)],
                has_substrate: hs, substrate_name: sn });
        }
    }}
    for &ci in &cys { for &hi in &his {
        if adist(&atoms[ci], &atoms[hi]) > 4.5 { continue; }
        for &di in &asp {
            if adist(&atoms[hi], &atoms[di]) > 4.5 { continue; }
            sites.push(ActiveSite { kind: "Cys-His-Asp".into(), rxn: RxnType::ProtonRelay.label().into(),
                barrier: RxnType::ProtonRelay.barrier_ev(),
                residues: vec![format!("CYS{}",atoms[ci].seq),format!("HIS{}",atoms[hi].seq),format!("ASP{}",atoms[di].seq)],
                has_substrate: false, substrate_name: String::new() });
        }
    }}

    // 7. Zn → Lewis Acid
    for &zi in &zn {
        let coord: Vec<String> = atoms.iter().filter(|a| !a.is_het && adist(&atoms[zi], a) < 3.0
            && matches!(a.name.as_str(),"NE2"|"ND1"|"SG"|"OD1"|"OD2"|"OE1"|"OE2"|"OG"))
            .map(|a| format!("{}{}:{}", a.res, a.seq, a.name)).collect();
        if coord.len() >= 2 {
            sites.push(ActiveSite { kind: "Zn Lewis acid".into(), rxn: RxnType::LewisAcid.label().into(),
                barrier: RxnType::LewisAcid.barrier_ev(), residues: coord,
                has_substrate: false, substrate_name: String::new() });
        }
    }

    // 8. 2His-1Carb facial triad — metal/2OG gated
    for (i, &h1) in his.iter().enumerate() { for &h2 in &his[i+1..] {
        if adist(&atoms[h1], &atoms[h2]) > 7.0 { continue; }
        if atoms[h1].chain != atoms[h2].chain { continue; }
        for &ei in asp.iter().chain(glu.iter()) {
            if adist(&atoms[h1], &atoms[ei]) < 5.5 && adist(&atoms[h2], &atoms[ei]) < 6.5 {
                let has_m = if v13 {
                    // v1.3: echte Facial Triad = BEIDE His und das Carboxylat koordinieren dasselbe katalytische Metall
                    fe.iter().chain(mn.iter()).chain(cu.iter()).chain(co.iter()).any(|&mi| cat[mi]
                        && adist(&atoms[mi],&atoms[h1])<LIG_CUT && adist(&atoms[mi],&atoms[h2])<LIG_CUT && adist(&atoms[mi],&atoms[ei])<LIG_CUT)
                } else {
                    fe.iter().chain(mn.iter()).chain(cu.iter()).chain(co.iter())
                    .any(|&mi| adist(&atoms[mi],&atoms[h1])<5.0||adist(&atoms[mi],&atoms[h2])<5.0||adist(&atoms[mi],&atoms[ei])<5.0)
                };
                let has_2og = og_at.iter().any(|&oi| adist(&atoms[oi],&atoms[h1])<8.0||adist(&atoms[oi],&atoms[h2])<8.0);
                let rxn = if has_m||has_2og { RxnType::RadicalCH } else { RxnType::ProtonRelay };
                let (hs,sn) = find_sub(h1, 6.0);
                let k = if has_m {"2His-1Carb (metal)"} else if has_2og {"2His-1Carb (2OG→Fe)"} else {"2His-1Carb"};
                sites.push(ActiveSite { kind: k.into(), rxn: rxn.label().into(), barrier: rxn.barrier_ev(),
                    residues: vec![format!("HIS{}",atoms[h1].seq),format!("HIS{}",atoms[h2].seq),
                        format!("{}{}",atoms[ei].res,atoms[ei].seq)],
                    has_substrate: hs||has_2og, substrate_name: sn });
            }
        }
    }}

    // 9. Tyr radical — metal/heme gated
    for &ti in &tyr { for &hi in &his {
        if adist(&atoms[ti], &atoms[hi]) > 5.0 { continue; }
        let m_near = fe.iter().chain(cu.iter()).any(|&mi| cat[mi] && adist(&atoms[ti],&atoms[mi])<5.0);
        // v1.3: Häm zählt nur, wenn sein Fe katalytisch ist (oder kein Fe aufgelöst wurde)
        let heme_ok = !v13 || fe.is_empty() || fe.iter().any(|&mi| cat[mi]
            && heme_at.iter().any(|&h| adist(&atoms[mi],&atoms[h])<4.5));
        let h_near = heme_ok && heme_at.iter().any(|&h| adist(&atoms[ti],&atoms[h])<6.0);
        if !m_near && !h_near { continue; }
        let ts = atoms[ti].seq;
        if sites.iter().any(|s| (s.kind.contains("-oxo")||s.kind.contains("-heme"))
            && s.residues.iter().any(|r| r.contains(&format!("TYR{}",ts)))) { continue; }
        let (hs,sn) = find_sub(ti, 6.0);
        sites.push(ActiveSite { kind: if h_near{"Tyr radical (heme)"}else{"Tyr radical (metal)"}.into(),
            rxn: RxnType::RadicalCH.label().into(), barrier: RxnType::RadicalCH.barrier_ev()*0.9,
            residues: vec![format!("TYR{}",atoms[ti].seq),format!("HIS{}",atoms[hi].seq)],
            has_substrate: hs||h_near, substrate_name: if hs{sn}else if h_near{"HEM".into()}else{String::new()} });
    }}

    // 10. Heme without resolved Fe
    if fe.is_empty() && !heme_at.is_empty() {
        let hc = &atoms[heme_at[0]];
        let ph: Vec<String> = his.iter().filter(|&&hi| adist(hc, &atoms[hi]) < 6.0)
            .map(|&hi| format!("HIS{}:{}", atoms[hi].seq, atoms[hi].name)).collect();
        if !ph.is_empty() && !sites.iter().any(|s| s.kind.contains("heme")) {
            sites.push(ActiveSite { kind: format!("Heme-noFe ({})", atoms[heme_at[0]].res),
                rxn: RxnType::RadicalCH.label().into(), barrier: RxnType::RadicalCH.barrier_ev(),
                residues: ph, has_substrate: true, substrate_name: atoms[heme_at[0]].res.clone() });
        }
    }

    // Deduplicate
    sites.sort_by(|a,b| b.barrier.partial_cmp(&a.barrier).unwrap_or(std::cmp::Ordering::Equal));
    let mut dd: Vec<ActiveSite> = Vec::new();
    for s in &sites {
        let mut k: Vec<String> = s.residues.clone(); k.sort();
        if !dd.iter().any(|d| { let mut dk: Vec<String> = d.residues.clone(); dk.sort(); dk == k }) { dd.push(s.clone()); }
    }
    dd
}

// ═══ APSU SCORE + KIE CLASS ESTIMATE ═══════════════════════════════════════
// Score 0-100 from: class prior (Lindblad, fixed per motif type) + motif counts + substrate.
// Decision threshold as in the frozen benchmarks.
pub const SCORE_THRESHOLD: f64 = 10.0;
#[derive(Clone, Debug, Serialize)]
struct QVS { total: f64, max_enhancement: f64, primary_rxn: String, predicted_kie: f64,
    n_radical: usize, n_hydride: usize, n_relay: usize, n_lewis: usize,
    has_substrate: bool, class: String, warning: String }

fn compute_qvs(sites: &[ActiveSite], temp: f64) -> QVS {
    let nr = sites.iter().filter(|s| s.rxn.contains("Radical")).count();
    let nh = sites.iter().filter(|s| s.rxn.contains("Hydride")).count();
    let nrl = sites.iter().filter(|s| s.rxn.contains("Relay")).count();
    let nl = sites.iter().filter(|s| s.rxn.contains("Lewis")).count();
    let hs = sites.iter().any(|s| s.has_substrate);
    let best = sites.iter().max_by(|a,b| a.barrier.partial_cmp(&b.barrier).unwrap_or(std::cmp::Ordering::Equal));
    let (pr, pb) = match best { Some(s) => (s.rxn.clone(), s.barrier), None => ("None".into(), 0.0) };
    let me = if pb > 0.0 {
        let d = if pr.contains("Radical"){1.2} else if pr.contains("Hydride"){1.1} else if pr.contains("Relay"){0.5} else {0.8};
        lindblad_enh(pb, d+1.0, temp)
    } else { 1.0 };

    let enh_sc = if me > 1.0 { (50.0 * me.log10() / 8.0).max(0.0).min(50.0) } else { 0.0 };
    let rad_sc = ((nr as f64 * 8.0) + (nh as f64 * 4.0)).min(25.0);
    let rel_pen = if nr==0&&nh==0 { (nrl as f64*8.0+nl as f64*4.0).min(30.0) } else { (nrl as f64*2.0).min(10.0) };
    let sub_b = if hs&&(nr>0||nh>0) { 10.0 } else if hs { 3.0 } else { 0.0 };
    let mut total = (enh_sc + rad_sc + sub_b - rel_pen).max(0.0).min(100.0);
    if nr==0 && nh==0 && nrl>0 { total = total.min(15.0); }

    // v1.4: KIE-Schätzung pro Reaktionsklasse = geometrisches Mittel der DEV-Literaturwerte
    // (ohne Ausschlüsse RNR/COMT). Innerhalb einer Klasse korreliert QVS NICHT mit dem KIE
    // (DEV: Spearman -0,41 radikal, -0,13 Hydrid), daher kein QVS-Term mehr.
    // Typischer Fehler (Leave-one-out): Faktor ~1,8. Aussage: Größenordnung, keine Rangfolge.
    let predicted_kie = if total < 8.0 { 1.0 } else if nr > 0 { 16.5 } else if nh > 0 { 5.0 } else { 1.0 };

    // v1.5: classes aligned with the benchmark decision (score >= SCORE_THRESHOLD = positive).
    let _ = me;
    let (class, warning) = if total >= 40.0 {
        ("TUNNELLING-LIKELY".into(), "Strong catalytic H-transfer site. Candidate for a KIE measurement; model with QM/MM including tunnelling.".to_string())
    } else if total >= SCORE_THRESHOLD {
        ("TUNNELLING-POSSIBLE".into(), "Catalytic H-transfer site present. A KIE measurement would decide.".to_string())
    } else {
        ("TUNNELLING-UNLIKELY".into(), "No catalytic H-transfer site found.".to_string())
    };

    QVS { total, max_enhancement: me, primary_rxn: pr, predicted_kie,
        n_radical: nr, n_hydride: nh, n_relay: nrl, n_lewis: nl,
        has_substrate: hs, class, warning }
}

fn full_scan(pdb: &str) -> (String, usize, Vec<ActiveSite>, QVS, f64) {
    let t0 = Instant::now(); let (t, a) = parse_pdb(pdb); let na = a.len();
    let s = classify_sites(&a); let q = compute_qvs(&s, 310.15);
    (t, na, s, q, t0.elapsed().as_secs_f64()*1000.0)
}

// ═══ DIAGNOSTIC ═════════════════════════════════════════════════════════
#[allow(dead_code)]
fn pdb_diag(pdb: &str, name: &str) {
    let (_, atoms) = parse_pdb(pdb);
    let metals: Vec<String> = atoms.iter().filter(|a| matches!(a.elem.as_str(),"FE"|"CU"|"MN"|"ZN"|"CO"|"MO"|"NI"|"MG"))
        .map(|a| format!("{}({}{},het={})", a.elem, a.res, a.seq, a.is_het)).collect();
    let mut h: Vec<String> = atoms.iter().filter(|a| a.is_het).map(|a| a.res.clone()).collect();
    h.sort(); h.dedup();
    eprintln!("    \x1b[35m│DIAG {}│ Met:{} │ HET:{} │ hem:{} qui:{} fol:{} nad:{} 2og:{} pte:{} sam:{} nuc:{}\x1b[0m",
        name, if metals.is_empty(){"∅".into()}else{metals.join(",")}, h.join(","),
        atoms.iter().filter(|a| is_heme(&a.res)).count(), atoms.iter().filter(|a| is_quinone(&a.res)).count(),
        atoms.iter().filter(|a| is_folate(&a.res)).count(), atoms.iter().filter(|a| is_nad_fad(&a.res)).count(),
        atoms.iter().filter(|a| is_2og(&a.res)).count(), atoms.iter().filter(|a| is_pterin(&a.res)).count(),
        atoms.iter().filter(|a| is_sam(&a.res)).count(), atoms.iter().filter(|a| is_nucleotide_substrate(&a.res)).count());
}

// ═══ TARGETS — 65 enzymes with published KIE values ═════════════════════
// Sources: Klinman (2014), Scrutton (2015), Kohen (2003), Schwartz (2009)
struct Target { name: &'static str, pdb: &'static str, kie: f64 }
fn pos() -> Vec<Target> { vec![
    // ─── RADICAL C-H (KIE > 5) ─── Literature: Klinman, Krebs, Bollinger
    Target{name:"SLO-1",pdb:"1YGE",kie:81.0},        // Soybean lipoxygenase (Klinman 1994)
    Target{name:"15-LOX",pdb:"1LOX",kie:40.0},        // 15-lipoxygenase (Glickman 1994)
    Target{name:"AADH",pdb:"2AH1",kie:55.0},          // Aromatic amine DH (Scrutton 2004)
    Target{name:"TauD",pdb:"1OS7",kie:37.0},          // Taurine dioxygenase (Bollinger 2003)
    Target{name:"COX-2",pdb:"3HS5",kie:27.0},         // v1.2: 3HS5 has heme+Fe+substrate
    Target{name:"GalOx",pdb:"1GOF",kie:22.5},         // Galactose oxidase (Whittaker 1998)
    Target{name:"MADH",pdb:"2BBK",kie:16.8},          // Methylamine DH (Scrutton 1999)
    Target{name:"CYP3A4",pdb:"1TQN",kie:11.5},        // Cytochrome P450 3A4
    Target{name:"P450cam",pdb:"2CPP",kie:11.0},       // P450cam (Groves 1978)
    Target{name:"P450BM3",pdb:"1BU7",kie:10.0},       // P450 BM3 (Munro 1996)
    Target{name:"PAM",pdb:"1PHM",kie:10.6},           // Peptidylglycine α-amidating
    Target{name:"MAO-B",pdb:"1GOS",kie:9.2},          // Monoamine oxidase B
    Target{name:"MAO-A",pdb:"1O5W",kie:8.0},          // Monoamine oxidase A
    Target{name:"PHase",pdb:"1PAH",kie:6.0},          // Phenylalanine hydroxylase (Fitzpatrick)
    Target{name:"XO",pdb:"1FIQ",kie:7.0},             // Xanthine oxidase (Hille 2005)
    Target{name:"MMO",pdb:"1MTY",kie:50.0},           // Methane monooxygenase (Lippard)
    // ─── HYDRIDE TRANSFER (KIE 3-15) ─── Literature: Kohen, Hay, Scrutton
    Target{name:"ECAO",pdb:"1OAC",kie:12.3},          // E. coli copper amine oxidase
    Target{name:"MR",pdb:"1GWJ",kie:15.4},            // Morphinone reductase (Scrutton 2006)
    Target{name:"P4H",pdb:"1MZE",kie:8.0},            // v1.1: 1MZE has Fe (was 1GNZ)
    Target{name:"PETNR",pdb:"1H50",kie:7.8},          // PETN reductase (Scrutton 2005)
    Target{name:"KDM4A",pdb:"2OQ6",kie:6.5},          // Jumonji demethylase
    Target{name:"Aconitase",pdb:"1AMJ",kie:6.0},      // Iron-sulfur enzyme
    Target{name:"TMADH",pdb:"1DJN",kie:4.6},          // Trimethylamine DH (Scrutton 2002)
    Target{name:"TS",pdb:"2TSC",kie:4.0},             // v1.1: 2TSC has substrate (was 1HZW)
    Target{name:"LADH",pdb:"1LDE",kie:3.8},           // Liver alcohol DH (Klinman 1981)
    Target{name:"DHFR",pdb:"1RX2",kie:3.5},           // Dihydrofolate reductase (Kohen 2004)
    Target{name:"ADH",pdb:"1HSO",kie:3.2},            // Alcohol dehydrogenase
    Target{name:"LDH",pdb:"1I10",kie:3.2},            // Lactate dehydrogenase (Kohen 2011)
    Target{name:"GluDH",pdb:"1HWZ",kie:3.0},          // Glutamate dehydrogenase
    Target{name:"DHODH",pdb:"1D3G",kie:5.0},          // Dihydroorotate DH (Malmquist 2008)
    Target{name:"G6PD",pdb:"1QKI",kie:3.5},           // Glucose-6-phosphate DH
    Target{name:"ACOMD",pdb:"3MDE",kie:4.0},          // Acyl-CoA dehydrogenase
    Target{name:"FLb2",pdb:"1FCB",kie:5.0},           // Flavocytochrome b2 (Scrutton)
    Target{name:"TrxR",pdb:"1TDE",kie:3.0},           // Thioredoxin reductase
    Target{name:"GR",pdb:"3GRS",kie:3.0},             // Glutathione reductase
    Target{name:"RNR",pdb:"1MXR",kie:4.0},            // v1.2: 1MXR R2 subunit with di-iron
    Target{name:"COMT",pdb:"3BWM",kie:4.5},           // Catechol O-methyltransferase
    Target{name:"GOx",pdb:"1CF3",kie:12.0},           // Glucose oxidase (Roth 2004)
    Target{name:"HRP",pdb:"1HCH",kie:3.0},            // Horseradish peroxidase
]}
fn neg() -> Vec<Target> { vec![
    // ─── CLASSICAL (KIE ≈ 1.0) ─── No tunneling contribution
    Target{name:"Lysozyme",pdb:"1LYZ",kie:1.0},
    Target{name:"Trypsin",pdb:"1TRN",kie:1.2},
    Target{name:"CA-II",pdb:"1CA2",kie:1.5},
    Target{name:"Chymotrypsin",pdb:"4CHA",kie:1.1},
    Target{name:"RNase A",pdb:"7RSA",kie:1.3},
    Target{name:"Thermolysin",pdb:"1TLX",kie:1.0},
    Target{name:"TIM",pdb:"1TIM",kie:1.4},
    Target{name:"Subtilisin",pdb:"1SBT",kie:1.1},
    Target{name:"Pepsin",pdb:"4PEP",kie:1.0},
    Target{name:"PLA2",pdb:"1BP2",kie:1.2},
    Target{name:"Elastase",pdb:"3EST",kie:1.1},
    Target{name:"Papain",pdb:"9PAP",kie:1.3},
    Target{name:"Thrombin",pdb:"1PPB",kie:1.0},
    Target{name:"AChE",pdb:"1ACJ",kie:1.2},
    Target{name:"ProteinaseK",pdb:"2PRK",kie:1.0},
    Target{name:"BetaLact",pdb:"1BTL",kie:1.0},       // β-lactamase
    Target{name:"Hexokinase",pdb:"1HKG",kie:1.0},
    Target{name:"Aldolase",pdb:"1ZAH",kie:1.0},
    Target{name:"Enolase",pdb:"2ONE",kie:1.0},
    Target{name:"PKA",pdb:"1ATP",kie:1.0},            // Protein kinase A
]}

fn download(id: &str, cache: &str) -> Result<String, String> {
    let p = format!("{}/{}.pdb", cache, id.to_uppercase());
    if let Ok(c) = fs::read_to_string(&p) { return Ok(c); }
    let url = format!("https://files.rcsb.org/download/{}.pdb", id.to_uppercase());
    let out = std::process::Command::new("curl").args(&["-sS","-f","--max-time","30",&url])
        .output().map_err(|e| format!("curl: {}", e))?;
    if out.status.success() { let t = String::from_utf8_lossy(&out.stdout).to_string();
        let _ = fs::write(&p, &t); Ok(t) } else { Err(format!("HTTP: {}", id)) }
}
// ═══ CLI ═════════════════════════════════════════════════════════════════
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn cli_scan(path: &str, json: bool) {
    let c = match fs::read_to_string(path) { Ok(c)=>c, Err(e)=>{eprintln!("Error:{}",e);return;} };
    let (title,na,sites,qvs,ms) = full_scan(&c);
    if json {
        #[derive(Serialize)] struct R{title:String,atoms:usize,sites:Vec<ActiveSite>,score:f64,class:String,predicted_kie:f64,class_prior:f64,primary_rxn:String,note:String,ms:f64}
        let r=R{title,atoms:na,sites,score:qvs.total,class:qvs.class,predicted_kie:qvs.predicted_kie,class_prior:qvs.max_enhancement,primary_rxn:qvs.primary_rxn,note:qvs.warning,ms};
        println!("{}",serde_json::to_string_pretty(&r).unwrap_or_default());
    } else {
        println!("\n  MEG-APSU v{} — {}",VERSION,title);
        println!("  {} atoms · {} sites",na,sites.len());
        for s in &sites {
            let sub=if s.has_substrate{format!(" [{}]",s.substrate_name)}else{String::new()};
            println!("    {} — {} (class barrier {:.2} eV){}",s.kind,s.rxn,s.barrier,sub);
        }
        println!("  APSU score: {:.1}/100 [{}]  KIE class estimate ≈{:.1}",qvs.total,qvs.class,qvs.predicted_kie);
        println!("  Class prior: {:.3e} ({}; fixed per motif type, not computed from coordinates)",qvs.max_enhancement,qvs.primary_rxn);
        println!("  {}\n  {:.0}ms\n",qvs.warning,ms);
    }
}
fn batch_scan(dir: &str, json: bool) {
    let t0=Instant::now();
    let files:Vec<_>=match fs::read_dir(dir){Ok(e)=>e.filter_map(|e|e.ok()).collect(),Err(e)=>{eprintln!("Error:{}",e);return;}};
    let pdbs:Vec<_>=files.iter().filter(|e|e.path().extension().map(|x|x=="pdb").unwrap_or(false)).collect();
    if !json{eprintln!("\n  \x1b[36m\x1b[1m═══ MEG-APSU v{} — BATCH ({} files) ═══\x1b[0m\n",VERSION,pdbs.len());}
    #[derive(Serialize)] struct BR{file:String,title:String,score:f64,class:String,kie:f64,rxn:String,sites:usize}
    let mut res:Vec<BR>=Vec::new(); let(mut nl,mut np,mut nu)=(0,0,0);
    for(i,e)in pdbs.iter().enumerate(){
        let p=e.path(); let f=p.file_name().unwrap_or_default().to_string_lossy().to_string();
        if let Ok(pdb)=fs::read_to_string(&p){
            let(t,_,s,q,ms)=full_scan(&pdb);
            if !json{let m=if q.total>=40.0{"\x1b[31m●\x1b[0m"}else if q.total>=SCORE_THRESHOLD{"\x1b[33m●\x1b[0m"}else{"\x1b[32m○\x1b[0m"};
                eprintln!("  [{:4}/{}] {} {:5.1} [{:20}] KIE≈{:5.1} {:.0}ms {}",i+1,pdbs.len(),m,q.total,q.class,q.predicted_kie,ms,f);}
            match q.class.as_str(){"TUNNELLING-LIKELY"=>{nl+=1},"TUNNELLING-POSSIBLE"=>{np+=1},_=>{nu+=1}}
            res.push(BR{file:f,title:t,score:q.total,class:q.class,kie:q.predicted_kie,rxn:q.primary_rxn,sites:s.len()});
        }
    }
    if json{println!("{}",serde_json::to_string_pretty(&res).unwrap_or_default());}
    else{let n=res.len();
        eprintln!("\n  \x1b[36m\x1b[1m═══ SUMMARY ═══\x1b[0m");
        eprintln!("    Total: {}  likely: {} ({:.1}%)  possible: {} ({:.1}%)  unlikely: {} ({:.1}%)",
            n,nl,nl as f64/n.max(1) as f64*100.0,np,np as f64/n.max(1) as f64*100.0,nu,nu as f64/n.max(1) as f64*100.0);
        eprintln!("    {:.1}s\n",t0.elapsed().as_secs_f64());
    }
}

fn main() {
    eprintln!("\x1b[36m  MEG-APSU v{} — structure-based triage of hydrogen-tunnelling enzymes", VERSION);
    eprintln!("  sectio-aurea-q · MEGALODON Research\x1b[0m\n");
    let args:Vec<String>=env::args().collect();
    let json=args.iter().any(|a|a=="--json");
    match args.get(1).map(|s|s.as_str()).unwrap_or("help") {
        "bench"=>bench::bench(args.get(2).map(|s|s.as_str()).unwrap_or("")),
        "scan"=>if let Some(f)=args.get(2){cli_scan(f,json);}else{eprintln!("  meg-apsu scan <pdb> [--json]");},
        "batch"=>if let Some(d)=args.get(2){batch_scan(d,json);}else{eprintln!("  meg-apsu batch <dir> [--json]");},
        _=>{eprintln!("  meg-apsu scan <pdb> [--json]     one structure");
            eprintln!("  meg-apsu batch <dir> [--json]    all .pdb files in a directory");
            eprintln!("  meg-apsu bench dev|test|test14|test15  reproduce the benchmarks (frozen test sets)");}
    }
}

// ═══ HELD-OUT BLIND TEST SET ════════════════════════════════════════════
// 30 enzymes NOT used during algorithm development. Zero tuning allowed.
// Sources: Klinman (2014 review), Scrutton (2015), Kohen (2011), PMC reviews
fn held_pos() -> Vec<Target> { vec![
    // ─── RADICAL / HIGH KIE ─── None of these were in training
    Target{name:"5-LOX",pdb:"3O8Y",kie:23.0},         // 5-lipoxygenase (Klinman)
    Target{name:"12-LOX",pdb:"3D3L",kie:30.0},        // 12-lipoxygenase (human platelet)
    Target{name:"TyrH",pdb:"2TOH",kie:5.0},           // Tyrosine hydroxylase (Fitzpatrick 2003)
    Target{name:"TrpH",pdb:"1MLW",kie:4.0},           // Tryptophan hydroxylase (McKinney 2001)
    Target{name:"CYP2D6",pdb:"2F9Q",kie:9.0},         // Cytochrome P450 2D6
    Target{name:"CYP2C9",pdb:"1OG5",kie:8.0},         // Cytochrome P450 2C9
    Target{name:"IDO",pdb:"2D0T",kie:6.0},            // Indoleamine 2,3-dioxygenase (heme)
    Target{name:"CPO",pdb:"1CPO",kie:5.0},            // Chloroperoxidase (heme-thiolate)
    // ─── HYDRIDE / MODERATE KIE ───
    Target{name:"NAO",pdb:"2C0U",kie:9.2},            // Nitroalkane oxidase (Gadda 2008, FAD)
    Target{name:"ChoOx",pdb:"2JBV",kie:10.0},         // Choline oxidase (Gadda 2005, FAD)
    Target{name:"PHBH",pdb:"1PBE",kie:3.5},           // p-Hydroxybenzoate hydroxylase (FAD)
    Target{name:"ICDH",pdb:"1AI2",kie:3.0},           // Isocitrate dehydrogenase (NAD)
    Target{name:"MDH",pdb:"4WLU",kie:3.0},            // Malate dehydrogenase (NAD+malate holo)
    Target{name:"FDH",pdb:"2NAD",kie:3.0},            // Formate dehydrogenase (NAD)
    Target{name:"DHPR",pdb:"1DHR",kie:3.0},           // Dihydropterin reductase (NAD)
    Target{name:"SDH",pdb:"1NEK",kie:4.0},            // Succinate dehydrogenase (FAD, Fe-S)
]}
fn held_neg() -> Vec<Target> { vec![
    // ─── CLASSICAL ─── None of these were in training
    Target{name:"PyrKin",pdb:"1PKN",kie:1.0},         // Pyruvate kinase
    Target{name:"PFK",pdb:"3PFK",kie:1.0},            // Phosphofructokinase
    Target{name:"GlyPhos",pdb:"1GPB",kie:1.0},        // Glycogen phosphorylase
    Target{name:"CarbPepA",pdb:"5CPA",kie:1.0},       // Carboxypeptidase A (Zn)
    Target{name:"Lipase",pdb:"1LPB",kie:1.0},         // Pancreatic lipase
    Target{name:"DNaseI",pdb:"3DNI",kie:1.0},         // DNase I
    Target{name:"CA-I",pdb:"1HCB",kie:1.0},           // Carbonic anhydrase I (Zn)
    Target{name:"AmpC",pdb:"1FR1",kie:1.0},           // β-lactamase class C
    Target{name:"Urease",pdb:"1FWJ",kie:1.0},         // Urease (Ni, but hydrolysis)
    Target{name:"Lysozyme2",pdb:"1HEW",kie:1.0},      // Hen egg white lysozyme (different PDB)
    Target{name:"Amylase",pdb:"1PPI",kie:1.0},        // α-amylase
    Target{name:"Caspase3",pdb:"1PAU",kie:1.0},       // Caspase-3
    Target{name:"MMP9",pdb:"1GKC",kie:1.0},           // Matrix metalloprotease 9 (Zn)
    Target{name:"PPase",pdb:"1FAJ",kie:1.0},          // Inorganic pyrophosphatase
]}
