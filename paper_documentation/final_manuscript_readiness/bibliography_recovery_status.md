# Bibliography recovery status

## Search performed

Searched the repository outside `.git`, `.venv`, and the generated documentation body for BibTeX/BibLaTeX (`.bib`, `.bbl`, `.bst`), citation interchange (`.ris`, `.enw`, `.nbib`, `.csl`), manuscript source (`.tex`), reference-named files, compressed archives, and likely paper/document files. Also searched project Markdown, Python, notebooks, and reports for DOI, arXiv, bibliography, reference, and citation markers; inspected the project ZIP listing and extracted text/searchable contents of the available PDF project notes.

The repository has no source manuscript or original bibliography file. There are generated IEEE table `.tex` fragments in `paper_documentation/tables/`, but these are not a manuscript or bibliography. `paper_documentation_package.zip` contains the generated documentation package, not an archived paper bibliography. The two `NCLT_RealWorld_Project_Note.pdf` files are byte-identical project notes; text search found no references/bibliography section. Other PDFs are result plots, not literature sources. `configs/paper_v4_mamba_improvement/baseline_reference.json` is an experiment configuration, not a bibliographic record. No actual literature item with sufficiently complete metadata was recovered.

## References that must be supplied

No titles, authors, venues, years, or DOIs can be named from the available repository without guessing. Before manuscript submission, supply the following source material:

| Required material | What must be supplied |
|---|---|
| Original manuscript reference list | The actual bibliography file or manuscript archive used for the paper, including every cited item and its citation key. |
| Complete metadata per cited work | Full author list, exact title, venue/publisher, year, volume/issue/pages or article number where applicable, and DOI or stable source URL where one exists. Mark unavailable fields explicitly rather than inferring them. |
| Citation-to-claim mapping | Manuscript section/sentence or method supported by each citation, so related-work and method claims can be audited. |
| Sources for literature claims retained in the revised paper | The author-selected sources that support statements about selective state-space/Mamba models, learned filtering/KalmanNet, classical EKF/UKF tracking, and radar range-bearing tracking, if those topics are discussed as prior work. These are topic-level requirements only; this report does not select or invent papers. |
| Provenance for any adopted model or dataset description | Any publication or technical source the authors actually relied on for the simulator, observation/noise assumptions, baseline implementation, or architecture, if applicable. |

The existing `paper_documentation/literature/reference_registry.csv` and `citation_usage_matrix.csv` are empty templates, not recovered bibliographies. Manuscript related-work coverage and novelty positioning therefore remain unavailable until the authors provide verified sources.
