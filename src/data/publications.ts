export type Publication = {
  title: string;
  authors: string[];
  venue: string;
  year: number;
  type: "conference" | "journal" | "thesis";
  abbr?: string;
  doi?: string;
  url?: string;
  image?: string;
  imageAlt?: string;
  featured?: boolean;
};

export const publications: Publication[] = [
  {
    title:
      "Continuous SUN (Stable, Unique, and Novel) Metric for Generative Modeling of Inorganic Crystals",
    authors: ["Masahiro Negishi", "Hyunsoo Park", "Kinga O. Mastej", "Aron Walsh"],
    venue: "IOPscience Machine Learning: Science and Technology",
    year: 2026,
    type: "journal",
    abbr: "IOP-MLST",
    url: "https://iopscience.iop.org/article/10.1088/2632-2153/ae7d85",
    image: "/assets/img/entries/work/csun.pdf",
    featured: true,
  },
  {
    title:
      "Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals",
    authors: ["Masahiro Negishi", "Hyunsoo Park", "Kinga O. Mastej", "Aron Walsh"],
    venue: "AI for Accelerated Materials Design - NeurIPS 2025",
    year: 2025,
    type: "conference",
    abbr: "NeurIPS",
    url: "https://openreview.net/forum?id=PiKMmLHbEH",
  },
  {
    title: "WILTing Trees: Interpreting the Distance Between MPNN Embeddings",
    authors: ["Masahiro Negishi", "Thomas Gartner", "Pascal Welke"],
    venue: "ICML 2025",
    year: 2025,
    type: "conference",
    abbr: "ICML",
    url: "https://openreview.net/forum?id=lflqQWP1jy",
    image: "/assets/img/entries/work/wilt.png",
    featured: true,
  },
  {
    title:
      "Weakly Supervised Disentanglement from Distance-based Supervision",
    authors: ["Masahiro Negishi", "Masashi Sugiyama"],
    venue:
      "Master Thesis, Department of Information Science, School of Science, The University of Tokyo",
    year: 2025,
    type: "thesis",
  },
  {
    title:
      "Two-Stage Coefficient Estimation in Symbolic Regression for Scientific Discovery",
    authors: [
      "Masahiro Negishi",
      "Ryo Igarashi",
      "Yoshitaka Ushiku",
      "Yoshitomo Matsubara",
      "Naoya Chiba",
    ],
    venue: "Machine Learning and the Physical Sciences Workshop - NeurIPS 2024",
    year: 2024,
    type: "conference",
    abbr: "NeurIPS",
    url: "https://neurips.cc/virtual/2024/100013",
    image: "/assets/img/entries/work/symbolic.pdf",
    featured: true,
  },
  {
    title: "Scaling Laws of Dataset Size for VideoGPT",
    authors: [
      "Masahiro Negishi",
      "Makoto Sato",
      "Ryosuke Unno",
      "Koudai Tabata",
      "Taiju Watanabe",
      "Junnosuke Kamohara",
      "Taiga Kume",
      "Ryo Okada",
      "Yusuke Iwasawa",
      "Yutaka Matsuo",
    ],
    venue: "Proceedings of the Annual Conference of JSAI",
    year: 2023,
    type: "journal",
    abbr: "JSAI",
    url: "https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2G6OS21f05/_article/-char/en",
  },
  {
    title: "Construction and Validation of Action-Conditioned VideoGPT",
    authors: [
      "Koudai Tabata",
      "Junnosuke Kamohara",
      "Ryosuke Unno",
      "Makoto Sato",
      "Taiju Watanabe",
      "Taiga Kume",
      "Masahiro Negishi",
      "Ryo Okada",
      "Yusuke Iwasawa",
      "Yutaka Matsuo",
    ],
    venue: "Proceedings of the Annual Conference of JSAI",
    year: 2023,
    type: "journal",
    abbr: "JSAI",
    url: "https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_1G4OS21a02/_article/-char/en",
  },
  {
    title: "Scaling Laws of Model Size for World Models",
    authors: [
      "Makoto Sato",
      "Ryosuke Unno",
      "Masahiro Negishi",
      "Koudai Tabata",
      "Taiju Watanabe",
      "Junnosuke Kamohara",
      "Taiga Kume",
      "Ryo Okada",
      "Yusuke Iwasawa",
      "Yutaka Matsuo",
    ],
    venue: "Proceedings of the Annual Conference of JSAI",
    year: 2023,
    type: "journal",
    abbr: "JSAI",
    url: "https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2G5OS21e02/_article/-char/en",
  },
  {
    title:
      "Pairwise-constraint Classification in Weakly Supervised Machine Learning: Risk-consistent Approach and Classifier-consistent Approach",
    authors: ["Masahiro Negishi", "Masashi Sugiyama"],
    venue:
      "Bachelor Thesis, Department of Information Science, School of Science, The University of Tokyo",
    year: 2023,
    type: "thesis",
  },
];
