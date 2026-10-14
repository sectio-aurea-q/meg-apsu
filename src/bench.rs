// ═══ MEG-APSU BENCHMARK (v1.3) ══════════════════════════════════════════
// Ehrlicher Prüfstand. Regeln:
//  1. DEV darf beliebig oft laufen und darf das Modell beeinflussen.
//  2. TEST ist nach Proteinfamilie getrennt und eingefroren (06.10.2026).
//     Er wird erst nach Abschluss der Entwicklung EINMAL ausgewertet.
//  3. Feste Schwelle QVS >= 10. Keine Optimierung auf den Bewertungsdaten.
//  4. Jede Modellversion wird gegen eine triviale Kofaktor-Baseline gemessen.
use crate::{download, full_scan, parse_pdb, is_heme, is_nad_fad, is_folate, is_pterin, is_sam, is_quinone, Target, V13};
use std::sync::atomic::Ordering;
use std::{env, fs};

const THRESHOLD: f64 = 10.0;

#[derive(Clone, Copy, PartialEq)]
enum Label { Tunneling, Classical, Control }

struct Case { name: &'static str, pdb: &'static str, kie: f64, label: Label }

// v1.4: Wie gut ist das Label belegt? (vorab festgelegte Regel, nicht nach Ergebnissen)
//  Stark: KIE >= 10 (über semiklassischem Maximum) ODER Tunneling-Nachweis per
//         Temperaturabhängigkeit / Swain-Schaad in der Literatur (Quellen vor Veröffentlichung mit DOI belegen)
//  Ausgeschlossen: RNR (1MXR = R2, H-Abstraktion liegt in R1), COMT (Methyltransfer, kein H-Transfer)
//  Unsicher: KIE < 10 ohne weiteren Nachweis; TIM/CA-II (echte Protonentransfers, "klassisch" unbelegt)
#[derive(Clone, Copy, PartialEq)]
enum Evidence { Stark, Unsicher, Ausgeschlossen }

fn evidence(c: &Case) -> Evidence {
    match c.label {
        Label::Control => Evidence::Stark,
        Label::Classical => match c.name { "TIM" | "CA-II" => Evidence::Unsicher, _ => Evidence::Stark },
        Label::Tunneling => match c.name {
            "RNR" | "COMT" => Evidence::Ausgeschlossen,
            "DHFR" | "TS" | "LADH" | "TMADH" | "PETNR" | "MR" => Evidence::Stark,
            _ if c.kie >= 10.0 => Evidence::Stark,
            _ => Evidence::Unsicher,
        },
    }
}

fn from_targets(v: Vec<Target>, l: Label) -> Vec<Case> {
    v.into_iter().map(|t| Case { name: t.name, pdb: t.pdb, kie: t.kie, label: l }).collect()
}

// Kontrollen: Redox-Proteine OHNE Katalyse (Transport, Speicherung, Elektronentransfer).
// Sie binden Häm/Fe/Cu/Flavin, können aber keine C-H-Bindung spalten → kein Tunneling.
// DEV-Familien: Globine, mitochondriales Cyt c, Rubredoxin, Transferrin
fn dev_controls() -> Vec<Case> { vec![
    Case{name:"Myoglobin",      pdb:"1MBN", kie:1.0, label:Label::Control},
    Case{name:"Haemoglobin",    pdb:"2HHB", kie:1.0, label:Label::Control},
    Case{name:"Aplysia-Mb",     pdb:"1MBA", kie:1.0, label:Label::Control},
    Case{name:"Cyt c (horse)",  pdb:"1HRC", kie:1.0, label:Label::Control},
    Case{name:"Cyt c (rice)",   pdb:"1CCR", kie:1.0, label:Label::Control},
    Case{name:"Rubredoxin",     pdb:"1IRO", kie:1.0, label:Label::Control},
    Case{name:"Transferrin",    pdb:"1A8E", kie:1.0, label:Label::Control},
]}
// TEST-Familien (EINGEFROREN, in der Entwicklung nie angesehen):
// Cupredoxine, Cyt b5, Cyt b562, Flavodoxin, Ferredoxine, Hämerythrin, Ferritine
fn test_controls() -> Vec<Case> { vec![
    Case{name:"Azurin",         pdb:"4AZU", kie:1.0, label:Label::Control},
    Case{name:"Plastocyanin",   pdb:"1PLC", kie:1.0, label:Label::Control},
    Case{name:"Cyt b5",         pdb:"1CYO", kie:1.0, label:Label::Control},
    Case{name:"Cyt b562",       pdb:"256B", kie:1.0, label:Label::Control},
    Case{name:"Flavodoxin",     pdb:"1FLV", kie:1.0, label:Label::Control},
    Case{name:"Ferredoxin-pl",  pdb:"1A70", kie:1.0, label:Label::Control},
    Case{name:"Ferredoxin-II",  pdb:"1FXD", kie:1.0, label:Label::Control},
    Case{name:"Hemerythrin",    pdb:"2HMQ", kie:1.0, label:Label::Control},
    Case{name:"Ferritin-H",     pdb:"2FHA", kie:1.0, label:Label::Control},
    Case{name:"Bacterioferritin",pdb:"1BCF",kie:1.0, label:Label::Control},
]}

fn dev_set() -> Vec<Case> {
    let mut v = from_targets(crate::pos(), Label::Tunneling);
    v.extend(from_targets(crate::neg(), Label::Classical));
    v.extend(dev_controls()); v
}
fn test_set() -> Vec<Case> {
    let mut v = from_targets(crate::held_pos(), Label::Tunneling);
    v.extend(from_targets(crate::held_neg(), Label::Classical));
    v.extend(test_controls()); v
}

// Triviale Baseline: "Bindet das Protein irgendeinen Redox-Kofaktor oder ein Redox-Metall?"
// Wenn ein Modell diese Baseline nicht schlägt, misst es nichts darüber hinaus.
fn baseline(pdb: &str) -> bool {
    let (_, atoms) = parse_pdb(pdb);
    atoms.iter().any(|a| matches!(a.elem.as_str(), "FE"|"CU"|"MN"|"CO"|"MO")
        || is_heme(&a.res) || is_nad_fad(&a.res) || is_folate(&a.res)
        || is_pterin(&a.res) || is_sam(&a.res) || is_quinone(&a.res))
}

// Wilson-Intervall (95 %) für einen Anteil k/n
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 { return (0.0, 0.0); }
    let (z, nf) = (1.96_f64, n as f64); let p = k as f64 / nf;
    let den = 1.0 + z*z/nf; let c = (p + z*z/(2.0*nf)) / den;
    let h = z * ((p*(1.0-p)/nf + z*z/(4.0*nf*nf)).sqrt()) / den;
    ((c-h).max(0.0), (c+h).min(1.0))
}
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a,&b| x[a].partial_cmp(&x[b]).unwrap());
    let mut r = vec![0.0; x.len()]; let mut i = 0;
    while i < idx.len() {
        let mut j = i; while j+1 < idx.len() && x[idx[j+1]] == x[idx[i]] { j += 1; }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j { r[idx[k]] = avg; } i = j + 1;
    }
    r
}
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y)); let n = x.len() as f64; if n < 3.0 { return 0.0; }
    let (mx, my) = (rx.iter().sum::<f64>()/n, ry.iter().sum::<f64>()/n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() { let (a,b) = (rx[i]-mx, ry[i]-my); sxy+=a*b; sxx+=a*a; syy+=b*b; }
    if sxx>0.0 && syy>0.0 { sxy/(sxx*syy).sqrt() } else { 0.0 }
}

#[derive(Clone)]
struct Row { name: &'static str, pdb: &'static str, label: Label, kie: f64, v12: f64, v13: f64, kie13: f64, base: bool, ev: Evidence, cls: &'static str }

fn run(set: &[Case]) -> Vec<Row> {
    let cache = format!("{}/.meg-apsu-pdb-cache", env::var("HOME").unwrap_or("/tmp".into()));
    let _ = fs::create_dir_all(&cache);
    let mut rows = Vec::new();
    for c in set {
        let pdb = match download(c.pdb, &cache) { Ok(p) => p, Err(e) => { eprintln!("  FAIL {} {}", c.pdb, e); continue; } };
        V13.store(false, Ordering::SeqCst); let (_,_,_,q12,_) = full_scan(&pdb);
        V13.store(true,  Ordering::SeqCst); let (_,_,_,q13,_) = full_scan(&pdb);
        rows.push(Row{ name:c.name, pdb:c.pdb, label:c.label, kie:c.kie, v12:q12.total, v13:q13.total,
            kie13:q13.predicted_kie, base: baseline(&pdb), ev: evidence(c),
            cls: if q13.n_radical > 0 { "radical" } else if q13.n_hydride > 0 { "hydride" } else { "other" } });
    }
    rows
}

fn score(rows: &[Row], pred: &dyn Fn(&Row) -> bool) -> (usize, usize, usize, usize) {
    let (mut tp, mut fnn, mut tn, mut fp) = (0,0,0,0);
    for r in rows {
        let p = pred(r); let truth = r.label == Label::Tunneling;
        match (truth, p) { (true,true)=>tp+=1, (true,false)=>fnn+=1, (false,false)=>tn+=1, (false,true)=>fp+=1 }
    }
    (tp, fnn, tn, fp)
}

fn report(title: &str, rows: &[Row]) {
    eprintln!("\n  ═══ {} ({} Proteine) ═══", title, rows.len());
    eprintln!("  {:16} {:5} {:10} {:>6} {:>6} {:>5}  {}", "Protein","PDB","Klasse","v1.2","v1.3","Base","");
    for r in rows {
        let l = match r.label { Label::Tunneling=>"Tunneling", Label::Classical=>"Klassisch", Label::Control=>"Kontrolle" };
        let truth = r.label == Label::Tunneling;
        let ok = |p: bool| if p == truth { " " } else { "✗" };
        eprintln!("  {:16} {:5} {:10} {:>5.1}{} {:>5.1}{} {:>4}{}", r.name, r.pdb, l,
            r.v12, ok(r.v12>=THRESHOLD), r.v13, ok(r.v13>=THRESHOLD),
            if r.base {"ja"} else {"nein"}, ok(r.base));
    }
    let models: [(&str, Box<dyn Fn(&Row)->bool>); 3] = [
        ("v1.2 (alt)",      Box::new(|r: &Row| r.v12 >= THRESHOLD)),
        ("v1.3 (neu)",      Box::new(|r: &Row| r.v13 >= THRESHOLD)),
        ("Kofaktor-Baseline",Box::new(|r: &Row| r.base)),
    ];
    eprintln!("\n  {:18} {:>9} {:>9} {:>9} {:>17}  Konfusion", "Modell","Sens.","Spez.","Genauigk.","95%-KI Genauigk.");
    for (n, f) in models.iter() {
        let (tp,fnn,tn,fp) = score(rows, f.as_ref());
        let tot = tp+fnn+tn+fp; let acc = (tp+tn) as f64/tot.max(1) as f64;
        let (lo,hi) = wilson(tp+tn, tot);
        eprintln!("  {:18} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}–{:>5.1}%  TP={} FN={} TN={} FP={}", n,
            tp as f64/(tp+fnn).max(1) as f64*100.0, tn as f64/(tn+fp).max(1) as f64*100.0,
            acc*100.0, lo*100.0, hi*100.0, tp, fnn, tn, fp);
    }
    // Nur Kontrollen: Wie viele Nicht-Enzyme werden fälschlich als Tunneling markiert?
    let ctrl: Vec<&Row> = rows.iter().filter(|r| r.label==Label::Control).collect();
    if !ctrl.is_empty() {
        let f12 = ctrl.iter().filter(|r| r.v12>=THRESHOLD).count();
        let f13 = ctrl.iter().filter(|r| r.v13>=THRESHOLD).count();
        eprintln!("\n  Kontrollen fälschlich positiv: v1.2 {}/{}  ·  v1.3 {}/{}", f12, ctrl.len(), f13, ctrl.len());
    }
    // KIE-Vorhersage nur auf Tunneling-Enzymen (Rangkorrelation)
    let pos: Vec<&Row> = rows.iter().filter(|r| r.label==Label::Tunneling).collect();
    let lit: Vec<f64> = pos.iter().map(|r| r.kie).collect();
    let pk: Vec<f64> = pos.iter().map(|r| r.kie13).collect();
    eprintln!("  KIE-Rangkorrelation v1.3 (Spearman ρ, n={}): {:.3}", pos.len(), spearman(&lit, &pk));
}

fn split_report(title: &str, rows: &[Row]) {
    let n = |e: Evidence| rows.iter().filter(|r| r.ev == e).count();
    eprintln!("\n  Label-Belege: stark {} · unsicher {} · ausgeschlossen {}",
        n(Evidence::Stark), n(Evidence::Unsicher), n(Evidence::Ausgeschlossen));
    report(&format!("{} – wie bisher", title), rows);
    let ohne: Vec<Row> = rows.iter().filter(|r| r.ev != Evidence::Ausgeschlossen).cloned().collect();
    report(&format!("{} – ohne Ausschlüsse", title), &ohne);
    let stark: Vec<Row> = rows.iter().filter(|r| r.ev == Evidence::Stark).cloned().collect();
    report(&format!("{} – nur starke Labels", title), &stark);
}

pub fn bench(which: &str) {
    match which {
        "dev" => { let r = run(&dev_set()); split_report("DEV-SET (Entwicklung)", &r); }
        "dev-csv" => {
            let r = run(&dev_set());
            let mut out = String::from("name,pdb,label,evidence,cls,qvs13,kie_lit,kie_pred\n");
            for x in &r {
                let l = match x.label { Label::Tunneling=>"T", Label::Classical=>"K", Label::Control=>"C" };
                let e = match x.ev { Evidence::Stark=>"stark", Evidence::Unsicher=>"unsicher", Evidence::Ausgeschlossen=>"aus" };
                out += &format!("{},{},{},{},{},{:.2},{:.2},{:.2}\n", x.name, x.pdb, l, e, x.cls, x.v13, x.kie, x.kie13);
            }
            let _ = fs::write("apsu_dev.csv", out); eprintln!("  apsu_dev.csv geschrieben");
        }
        "test" => {
            eprintln!("\n  ⚠ TEST-SET: eingefroren. Nur einmal nach Abschluss der Entwicklung auswerten.");
            let r = run(&test_set()); split_report("TEST-SET (eingefroren, nach Familie getrennt)", &r);
        }
        _ => eprintln!("  meg-apsu bench dev|test"),
    }
}
