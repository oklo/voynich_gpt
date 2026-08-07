# A falsification-first audit of Voynich text hypotheses

## Bottom line

I did not decipher the Voynich manuscript, and the available evidence does not
justify a translation.  The strongest conclusion supported by these
experiments is narrower but useful:

> The EVA transcription is highly structured, but it is a poor match for
> ordinary running prose if EVA glyphs are letters and EVA-delimited tokens are
> words.  Its strongest organization is local, morphological, page-specific,
> and line-position-specific; its cross-word order signal is much weaker than
> in the natural-language controls.

That substantially weakens plaintext and simple-substitution explanations.  It
does **not** prove that the manuscript is meaningless: a code, a lossy or verbose
cipher, a notation system, or an encoding whose spaces are not word boundaries
can hide ordinary linguistic statistics.  Of the broad hypotheses considered
here, a layout-conditioned generative procedure or a code/notation system fits
the observed anomalies better than ordinary encrypted prose.

There is one positive lead worth retaining.  I independently reproduce the
published Hebrew/abjad **word-shape** signal across two transcriptions.  A new
period- and domain-matched comparison strengthens it in a specific way:
Currier-B herbal folios are substantially closer than other Currier-B sections
to both the materia-medica book of a 1491 Hebrew medical printing and a short
catalog transcription from a circa-1500 northern-Italian Hebrew herbal.  This is
the first result here that is specific to the illustrated herbal section, not
merely generic Hebrew-looking word lengths.

It is still not a decipherment.  The feature is insensitive to character and
word order, and meaningless pattern-matched pseudotext receives the identical
language ranking.  Voynich's sequence structure also remains far less
Hebrew-like than its isolated word shapes.  A direct formula-sequence test below
now strongly rejects the simplest version—a stable substitution or consistent
word-anagramming of Hebrew practical-herbal prose.  The shape result justifies
retaining a Hebrew or Hebrew-mediated botanical/medical source as a narrow,
falsifiable lead, not treating it as a translation.

A preregistered test on the manuscript's most tempting semantic anchors—the
zodiac diagrams—also fails.  One global exact substitution key can force only
three of twelve diagrams to contain their correct Hebrew sign/month, or four
under a many-to-one homophonic relaxation.  Those fits are no better than
shuffled sign assignments, and keys trained on alternating signs recover zero
correct held-out diagrams.  Thus the Hebrew lead has now failed both a
domain-specific sequence prediction and an illustration-conditioned vocabulary
prediction under simple cipher families.

An explicit held-out mechanism tournament now strengthens the broader
conclusion.  A period-materia-medica Hebrew character model with a substitution
key and reading direction fitted on training Voynich scores 3.734 bits per
held-out unit.  A Voynich-native morphology model scores 1.928, and supplying
Currier stratum plus physical line position improves this to 1.852.  The result
holds in all five whole-quire folds.  A literal previous-word copy/mutation
channel adds almost nothing, so the evidence is specifically for predictable
Voynich morphology and layout coupling—not yet for a particular pseudotext
algorithm.

## What was wrong or fragile in the original analysis

The [2025 blog post](https://oklo.org/2025/03/02/the-voynich-manuscript/)
was a reasonable exploratory start, but several details prevent its results
from carrying the interpretive weight placed on them.

1. **The cleaning step deletes word boundaries.** Both notebooks apply
   `re.sub(r'<.*?>', '', line)`.  In IVTFF, `<->` and `<~>` are not disposable
   markup: they are drawing interruptions that imply apparent word spaces.
   The [IVTFF 2.0 specification](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf)
   states this explicitly.  The regex removes exactly 875 such boundaries,
   changing 37,919 apparent tokens into the reported 37,044 and joining words
   across drawings.

2. **The checked-in training workflow no longer runs.** `train.py` imports
   `model.py` and executes `configurator.py`; commit `73c8f1c` deleted both.
   There is no environment or dependency lock, and no checkpoint is present.
   The logs remain as evidence of a past run, but the clean clone cannot
   reproduce it.

3. **Raw EVA characters are not a neutral alphabet.** EVA is a Latin
   transliteration convention.  Common manuscript glyphs or compounds such as
   `ch`, `sh`, and benched-gallows sequences occupy several Latin codepoints.
   The empirical order-1 entropy changes from 2.112 bits per raw EVA codepoint
   to 2.238 bits per unit under one documented grouping sensitivity.  Neither
   representation is asserted to be the truth; the change shows that an
   entropy result is partly a transcription decision.

4. **Cross-corpus loss is not directly semantic.** A character model's loss
   depends on alphabet, orthography, word length, split, and capacity.  Low
   loss means predictability in that representation, not meaningfulness or
   meaninglessness.  Hawaiian in the repository is also very predictable, for
   example.  The more diagnostic comparison is how much real word order helps
   relative to a within-corpus shuffle under held-out evaluation.

5. **The original shuffling experiment had a useful idea but a weak
   interpretation.** The 1.26 versus 1.40-ish loss difference says that order
   carries some information.  It does not establish syntax, and it was trained
   on the corrupted boundaries.  The tests below retain this core idea while
   using explicit nulls and held-out groups.

## Corpus and methods

The primary analysis parses `IT2a-n.txt` as IVTFF rather than stripping angle
brackets.  It keeps page, quire, illustration, Currier, hand, locus, and line
metadata, treats all four IVTFF apparent-space forms as boundaries, and omits
uncertain/non-basic tokens from conservative tests.

| Quantity | IT2a/Takahashi |
|---|---:|
| Pages | 225 |
| Apparent tokens, all loci | 37,919 |
| Certain tokens, all loci | 37,759 |
| Certain paragraph tokens | 34,411 |
| Certain paragraph types | 7,069 |
| Hapax types | 4,791 |
| Paragraph lines | 4,117 |
| Drawing-implied spaces lost by the old cleaner | 875 |

The core null permutes words independently within each physical line.  This
preserves vocabulary, frequency, line lengths, and the bag of words on every
line while destroying within-line order.  Reported primary p-values use 499
permutations, so the minimum resolved p-value is 0.002.  Character models are
smoothed order-5 models evaluated either leave-one-quire-out or in contiguous
10-fold splits; the comparison is always the real stream versus the same words
shuffled within the held-out grouping.

The results were repeated against the independent Zandbergen-Landini ZL 2b
transcription (SHA-256
`c7ffff9e1f3ecbec174e234c04f056b2bec14f8d722726c456f108e2c7060db5`).

## Results: structured, but not ordinary prose-like

### Internal structure

| Statistic | Observed | Within-line null | z | permutation p |
|---|---:|---:|---:|---:|
| Adjacent-word MI, top-100 vocabulary | 0.3201 bits | 0.2393 | 32.1 | 0.002 |
| Word/line-position MI | 0.07540 bits | 0.00580 | 148.0 | 0.002 |
| Adjacent one-edit variant rate | 0.03433 | 0.03015 | 5.18 | 0.002 |
| Adjacent exact-repeat rate | 0.00964 | 0.00902 | 1.35 | 0.104 |
| Tokens in repeated bigrams | 0.2272 | 0.1778 | 25.0 | 0.002 |
| Tokens in repeated trigrams | 0.01049 | 0.00276 | 16.9 | 0.002 |
| Tokens in repeated four-grams | 0.000180 | 0.000025 | 3.18 | 0.030 |

This rejects an independent or globally shuffled word process.  It also refines
the common “self-copying” description: one-edit variants are locally enriched,
but exact adjacent repeats are not significant under the line-preserving null.

The exceptional result is line position.  Words are strongly selected by
whether they begin, continue, or end a physical line.  The four matched plain
texts have essentially zero line-position effect after being reshaped to the
same line lengths.  This confirms the long-known line phenomenon with an
explicit null, and makes a simple transcription of prose difficult to sustain.

### Cross-word information is unusually weak

The table reports the held-out order-5 loss improvement caused by restoring
word order.  Using an internal shuffle makes this much more comparable than raw
loss across different alphabets.

| Corpus / representation | Ordering gain (bits/unit) | Approx. bits/word |
|---|---:|---:|
| Voynich raw EVA, leave-quire-out | 0.0517 | 0.314 |
| Voynich grouped-EVA sensitivity, leave-quire-out | 0.0685 | 0.379 |
| Voynich raw EVA, contiguous folds | 0.0538 | 0.326 |
| Voynich grouped-EVA sensitivity, contiguous folds | 0.0740 | 0.409 |
| English | 0.2774 | 1.475 |
| French | 0.2751 | 1.566 |
| Tagalog | 0.4637 | 2.450 |
| Hawaiian | 0.3923 | 1.605 |

Repeated trigrams tell the same story.  Voynich has more than its shuffled null
(1.05% versus 0.28% of trigram instances), but far fewer than matched English
(7.72%), French (5.87%), Tagalog (24.31%), or Hawaiian (45.16%).  This is an
independent recovery of the central result in [Reddy and Knight
(2011)](https://aclanthology.org/W11-1511.pdf): language-like morphology and
topics coexist with remarkably weak word-sequence structure and strong line
effects.

### Page-level variation is real but confounded

Word identity is associated with illustration category (0.214 bits), assigned
hand (0.204 bits), and Currier A/B (0.150 bits).  These associations are not
independent: hand, subject, quire, Currier label, and manuscript order overlap.
They establish heterogeneous production regimes, not semantic topics by
themselves.  This is compatible both with subject-dependent language and with a
generator whose tables or habits change across sections.

The anchor audit finds 1,023 certain label tokens but 748 types, of which 634
are hapax.  Still, 605 label tokens (350 types) are also attested in paragraph
text.  Labels therefore do not look like a wholly separate alphabet or channel,
but they are sparse enough that post-hoc dictionary glossing has enormous
freedom.  This is why a proposed key must be frozen and tested against held-out
labels rather than judged from a few attractive matches.

### Independent-transcription check

On ZL 2b, the central values remain close: adjacent-word MI 0.3390 versus a
0.2424 null, line-position MI 0.07339 versus 0.00582, one-edit adjacency 0.03612
versus 0.03093, and raw/grouped order gains of about 0.39/0.43 bits per word in
leave-quire-out tests.  The result is therefore not a peculiarity of the
Takahashi transcription.

These findings sit between two strands of prior work.  Long-range clustering
and section vocabulary have been interpreted as language-like by
[Montemurro and Zanette (2013)](https://doi.org/10.1371/journal.pone.0066344),
[Amancio et al. (2013)](https://doi.org/10.1371/journal.pone.0067310), and more
recently [Layfield and Davis (2026)](https://doi.org/10.63744/2ezxpskcezq4).
Local similarity, line dependence, and symbol-reuse autocorrelation have been
used to support procedural generation by [Timm and Schinner
(2019)](https://doi.org/10.1080/01611194.2019.1596999), [Matlach et al.
(2022)](https://doi.org/10.1371/journal.pone.0260948), and [Steckley and
Steckley (2024)](https://arxiv.org/abs/2404.13069).  The present audit finds
both kinds of structure; the weak cross-word effect and extreme line effect are
the more discriminating observations.

## The Hebrew/abjad lead

[Hauer and Kondrak (2016)](https://aclanthology.org/Q16-1006.pdf) replace each
word with the sorted multiplicities of its symbols: `seems → (2,2,1)`.  The
feature survives monoalphabetic substitution and within-word anagramming.  On
their Currier-B sample, Hebrew was the nearest of 380 UDHR languages at distance
0.020.  Their produced Hebrew was not syntactically or semantically coherent,
and the paper explicitly allows that the result may be an artifact of
anagramming plus language models.

I reimplemented the feature from its definition, using 253 independently
loaded NLTK UDHR representations and equal-size bootstrap samples.

| Transcription / target | EVA representation | Hebrew rank | Distance | Next relevant rank |
|---|---|---:|---:|---|
| IT2a Currier B | raw codepoints | 1 | 0.0377 | Arabic 2 (0.0511) |
| IT2a Currier B | grouped sensitivity | 1 | 0.0261 | Arabic 2 (0.0465) |
| ZL 2b Currier B | raw codepoints | 1 | 0.0382 | Arabic 2 (0.0559) |
| ZL 2b Currier B | grouped sensitivity | 1 | 0.0271 | Arabic 2 (0.0512) |
| IT2a Currier A | grouped sensitivity | 2 | 0.0509 | Persian 1 (0.0474) |
| ZL 2b Currier A | grouped sensitivity | 3 | 0.0527 | Persian/Beti 1–2 |

Hebrew won all 200 matched Currier-B bootstrap trials for both representations
and both transcriptions.  This is a robust reproduction, not something to wave
away.

It is also easy to overinterpret:

- A length-only ablation still ranks Hebrew first.  Under raw EVA, conditioning
  on word length drops Hebrew's repetition-pattern rank to 11th (IT2a) or 12th
  (ZL).  Under the grouped sensitivity it returns to first.  The stronger claim
  therefore depends on how EVA sequences are turned into manuscript units.
- Reversing or randomly shuffling every word leaves the statistic unchanged.
  More decisively, I generated pseudowords with a fresh random symbol mapping
  for every token—there is no consistent cipher key, lexicon, or message—and
  preserved only each token's multiplicity pattern.  The complete language
  ranking is identical by construction.
- A sequence sanity check sharply separates shape from language.  Currier B
  gains only 0.40–0.50 bits/word from real order across the two transcriptions;
  Hebrew UDHR gains about 1.46, Arabic 2.05–2.13, Persian 1.86–2.02, and English
  2.36–2.51.  Corpus lengths differ, so these are descriptive rather than a
  formal rejection test, but the gap is large and in the wrong direction for
  readable Hebrew.

The right interpretation is: **Voynich word shapes are abjad-like under this
feature, especially in Currier B.**  The result does not identify Hebrew rather
than another short-word, low-repetition orthography, and it supplies no token
mapping or sentence translation.  A 2026 open Hebrew-cipher exploration reached
the same practical failure mode: apparent dictionary matches concentrated in
short words, common Hebrew function-word behavior was absent, and syntax was
indistinguishable from random; its author publicly [withdrew the
hypothesis](https://www.reddit.com/r/voynich/comments/1s4cco3/i_spent_weeks_chasing_a_hebrew_cipher_hypothesis/).

### A preregistered Hebrew zodiac-anchor test

The zodiac sequence seems at first to offer ideal cribs: the central emblems
identify Pisces through Sagittarius.  Image inspection and the transcription
schema impose an important correction.  The readable words next to the
emblems—*mars*, *aberil*, *may*, and so on—are later Romance-language additions.
The Voynich-script `Lz` loci are labels of the surrounding zodiac elements,
mostly the star-holding figures, rather than a transcription field identifying
a central sign-name label.  The [folio/sign table and later month
annotations](https://voynich.nu/writing.html#extra) and the [`Lz` locus
definition](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf) make this
distinction explicit.  There is therefore no honest one-token “Aries” crib.

I froze a weaker page-level test before fitting a key.  It uses all 295 wholly
certain `Lz` loci on the twelve surviving diagrams; it does not select a label
after examining a Hebrew candidate.  For each diagram the fixed candidates are
the conventional Hebrew sign and corresponding Hebrew month, both bare and
with `מזל`, `חדש`, or `חודש` prefixes.  Spelling variants include the
medieval `ארי`/`אריה` and `מאזנים` forms and common full/defective month
spellings.  The zodiac list is attested in Abraham ibn Ezra's twelfth-century
[*Beginning of Wisdom*](https://www.betemunah.org/abraham-ibn-ezras-introductions-to-astrologypdf.pdf).

The exact ordered-cipher model deliberately favors the hypothesis.  It ignores
spaces inside each `Lz` locus, permits either reading direction separately for
every label, tests raw and grouped EVA, and tests both a bijection and a
many-cipher-symbols-to-one-Hebrew-letter relaxation.  Hebrew final letters are
alternately collapsed with their medial forms or kept distinct.  A single key
must serve every matched page.  The null shuffles the ten semantic targets
among the ten diagram blocks while keeping the duplicate Aries and Taurus
pages together.  Held-out evaluation uses two folds of alternating signs in
zodiac order.

| Representation / exact mapping | Hebrew finals | Best full fit | Shuffled-target p | Correct held-out top-1 credit |
|---|---|---:|---:|---:|
| Grouped EVA, bijective | Collapsed | 3/12 | 0.768 | 0/12 |
| Grouped EVA, many-to-one | Collapsed | 4/12 | 0.708 | 0/12 |
| Raw EVA, bijective | Collapsed | 3/12 | 0.588 | 0/12 |
| Raw EVA, many-to-one | Collapsed | 3/12 | 1.000 | 0/12 |
| Grouped EVA, bijective | Distinct | 3/12 | 0.818 | 0/12 |
| Grouped EVA, many-to-one | Distinct | 4/12 | 0.670 | 0/12 |

All p-values use 499 non-identity diagram-block permutations.  The two omitted
raw/distinct sensitivities also fit 3/12 (p=0.660 and 1.000).  Held-out credit
is zero in all eight conditions, with upper-tail permutation p=1.000.  The
primary bijective fit produces three superficially attractive readings:
`ched → קשת` on Sagittarius, reversed `otal → סיון` on Gemini, and
reversed `otey → חשון` on Scorpio.  Their joint key fails nine diagrams and
predicts none; shuffled sign assignments fit at least as well in 383 of 499
trials, before the finite-sample correction.  This is a concrete example of
why a few short-word matches do not constitute a decipherment.

Arbitrary within-word anagramming is too flexible for these short labels to
supply a discriminating local crib.  Every one of the twelve correct pages has
at least one locally compatible anagram; depending on the page, there are 48 to
8,166 local injective keys after collapsing final forms.  Yet twelve-of-twelve
feasibility is also routine after shuffling the signs (p=0.902), and the sum of
log local-key counts has p=0.658.  Keeping final forms distinct raises one page
to 40,368 local keys and still gives no significant page/sign association
(p=0.114).  This does not prove that no global anagram key exists; it shows that
these labels cannot support one without independent constraints.  The herbal
formula-sequence test remains the stronger rejection of a consistent
word-anagrammed Hebrew practical text.

The result is negative but decisive within its scope: **the zodiac pages do not
contain a recoverable Hebrew sign/month vocabulary under exact substitution,
reversal, or the tested homophonic relaxation, and local anagram matches are
combinatorially saturated.**

## Period- and domain-matched Hebrew controls

The generic UDHR ranking leaves a crucial question unanswered: is Voynich merely
close to Hebrew in the aggregate, or is the illustrated herbal text especially
close to Hebrew botanical and pharmacological writing?  Two unusually good
controls permit a first test.

The primary control is the Hebrew edition of Avicenna's *Canon of Medicine*,
printed in Naples in 1491--92.  Its main Hebrew translation was completed by
Nathan ha-Me'ati in Rome in 1279.  Book II concerns materia medica and simple
drugs, including the actions, collection, and preservation of herbal remedies;
the other four books provide within-work controls for general medicine, organ
diseases, general diseases, and compound formulations.  The section definitions
come from the [University of Manchester catalog](https://www.digitalcollections.manchester.ac.uk/view/MS-MEDICAL-00023-00002-00014/2),
and the analyzed [Yale scan and Tesseract OCR](https://archive.org/details/4072969.med.yale.edu)
are public.

I extracted Hebrew words from fixed scan-leaf ranges, discarded OCR words below
50 confidence, normalized the five context-dependent Hebrew final forms, and
used the same symbol-multiplicity feature as above.  Books III and IV are
size-matched samples because those books are much longer.  The critical
Currier-B comparisons are:

| Voynich subsection | Book II materia medica | Book III sample | Book I | Book V | Book IV sample |
|---|---:|---:|---:|---:|---:|
| Herbal, Currier B | **0.01253** | 0.01616 | 0.01846 | 0.01849 | 0.01759 |
| Text only, Currier B | **0.02408** | 0.02697 | 0.03113 | 0.03360 | 0.03110 |
| Biological/balneological, Currier B | **0.02720** | 0.02892 | 0.03202 | 0.03576 | 0.03202 |
| Marginal stars, Currier B | **0.02835** | 0.03015 | 0.03756 | 0.03434 | 0.03636 |

Lower is closer.  Book II was the nearest Canon section in 166 of 200
equal-size token resamples of Currier-B herbal text.  More importantly, when
the direction of comparison is reversed, the herbal group was the Voynich
subsection nearest to Book II in all 200 within-Currier-B resamples.  Its
advantage is not purely word length: for all Currier B, Book III is marginally
closer on length alone (0.01363 versus 0.01432), while Book II wins because the
within-length repeated-symbol structure is closer (0.00492 versus 0.00732).

The result is stable to transcription choices.  As the minimum OCR confidence
is increased from 0 to 30, 50, and 70, the herbal-B/Book-II distance changes
from 0.01810 to 0.01476, 0.01253, and 0.01095; Book II remains the nearest
section each time.  Keeping Hebrew final forms distinct changes the confidence-50
distance only from 0.01253 to 0.01309.

The second control is more direct.  [BnF Hébreu 1199](https://portail.biblissima.fr/fr/ark:/43093/mdatafbbaeb08a16281c0a5cb1a611347d74b5909e19a)
is a Hebrew *Liber Plantis* dated circa 1500 and localized to northern Italy.
Its folios 1--66v contain plant plates, short medical descriptions, magical
recipes, and some Latin/Italian captions.  Scholarship identifies it as a
fifteenth-century alchemical herbal translated from Latin, with plant captions
in Hebrew and Latin or Italian in Hebrew transliteration
([*Medieval Mediterranean Pharmacology*, pp. 9--10](https://www.ncbi.nlm.nih.gov/books/NBK606146/pdf/Bookshelf_NBK606146.pdf)).
The [BnF catalog](https://archivesetmanuscrits.bnf.fr/ark:/12148/cc8082r)
provides a catalog transcription of 880 Hebrew tokens on folios 67v--68v,
avoiding OCR entirely.

| Voynich subsection | Distance to BnF Hébreu 1199 transcript |
|---|---:|
| Herbal, Currier B | **0.04528** |
| Biological/balneological, Currier B | 0.06378 |
| Text only, Currier B | 0.06926 |
| Marginal stars, Currier B | 0.09051 |
| Herbal, Currier A | **0.05143** |
| Pharmaceutical, Currier A | 0.08417 |

The herbal subsection won 199 of 200 resamples within Currier B and all 200
within Currier A.  The absolute distance is larger than for the much bigger
printed Canon corpus, as expected for a small sample; the within-stratum ranking
is the informative part.

This is a genuine positive signal, but its scope is narrow.  Illustration type,
Currier language, hand, and manuscript order are partly confounded.  The BnF
sample is only two transcribed folios, and the token bootstrap measures ranking
stability rather than a probability of Hebrew authorship.  Above all, a
symbol-multiplicity distribution cannot map a single Voynich token to a Hebrew
word.  The evidence now says **“a Hebrew-mediated herbal/medical hypothesis
deserves direct testing,” not “the manuscript is Hebrew.”**

### A sequence-level test of the Hebrew herbal prediction

The direct comparison also creates a much riskier prediction than a
word-shape distance.  Sivan Gottlieb's study of Hébreu 1199 places it in the
indirect branch of the northern-Italian “Alchemical Herbals” tradition, probably
related through Florence MS 106.  The codex has 133 plant illustrations and 120
plant texts; most texts were collected on seventeen pages at the end, while
twenty-eight were also placed next to illustrations.  Its 98-entry herbal
sequence is inherited from a Latin source, although the Hebrew scribe adapted
it for Jewish readers and made errors.  This is not generic Hebrew prose.  It
is highly formulaic practical writing: entries normally introduce an ailment,
say “take this herb,” give preparation and administration instructions, report
that the patient will be cured, and often end with gathering or habitat
information.  The study counts 282 medicinal and 45 magical uses, 38
tested/proven claims, and wine in 44 recipes
([Gottlieb 2023](https://doi.org/10.1017/9781009389792.007)).

I extracted sixteen complete, human-transcribed heading/description entries
from the BnF catalog, excluding editorial bracket expansions.  I then compared
their repeated word sequences with complete Voynich herbal-page paragraph
streams.  This tests a property preserved by a stable monoalphabetic
substitution.  As a sensitivity test, every word was also replaced by its
sorted multiset of characters (using grouped EVA units): a consistent
within-word anagram preserves that representation too.

| Corpus | Entries/pages | Tokens | Top-token share | Repeated bigram instances | Repeated trigram instances | Repeated four-gram instances |
|---|---:|---:|---:|---:|---:|---:|
| BnF Hébreu 1199 | 16 | 837 | 0.0573 | **0.2570** | **0.0758** | **0.0152** |
| Voynich herbal, Currier A | 95 | 7,865 | 0.0503 | 0.1345 | 0.0042 | 0.0000 |
| Voynich herbal, Currier B | 32 | 3,432 | 0.0224 | 0.0771 | 0.0018 | 0.0000 |

The contrast is not a small-corpus artifact in the direction favorable to the
result: even with four to nine times as much text, neither Voynich stratum has
a single repeated four-word sequence.  In 1,000 half-entry resamples with the
Voynich comparison matched to the exact number and length of Hebrew entries,
Hébreu 1199 had
the higher bigram and trigram recurrence in all 1,000 trials against both
Currier A and B.  For four-grams it was higher in 844/1,000 trials against A
and 845/1,000 against B; most remaining trials were zero-zero ties.  Sorting
the characters within every word changes the trigram rates only from 0.0042 to
0.0044 for A and not at all for B, and gives the same result in all 1,000
matched trigram trials.

The recurring Hebrew phrases show why: `מזה העשב` (“of/from this
herb”) occurs 25 times and `קח מזה העשב` (“take of/from this herb”)
nine times in just 837 tokens.  Currier-B herbal's most frequent exact trigram
occurs only twice in 3,432 tokens.  A deterministic substitution or consistent
word-anagramming scheme cannot erase this equality pattern.  It would require
additional machinery—context-dependent homophones, changing keys, nulls,
splitting/merging of words, or a very different source genre.

This is the strongest domain-sequence negative result for the Hebrew lead.  The positive
isolated-word shape match survives, but the sequence behavior expected from
the closest known Hebrew botanical control does not.  The two facts are
compatible if the word-shape match is typological or artifactual; they are hard
to reconcile with Voynich as a simply substituted or anagrammed translation
of practical Hebrew herbal recipes.

The same control sharpens the manuscript's line-position anomaly.  I merged
the 24 training and seven held-out catalog-grounded lines from folio 67v and
tested the association between word identity and first, middle, or last
position against shuffling words within each physical line.  The absolute
Hebrew MI is upward-biased by its small vocabulary sample, so the within-corpus
null—not its raw magnitude—is the comparison:

| Corpus | Lines | Tokens | Observed MI | Within-line null | z | permutation p |
|---|---:|---:|---:|---:|---:|---:|
| Hébreu 1199, aligned 67v lines | 31 | 281 | 0.39770 | 0.39939 | -0.06 | 0.518 |
| Voynich herbal, Currier A | 1,225 | 7,865 | 0.11032 | 0.02109 | 43.32 | 0.001 |
| Voynich herbal, Currier B | 377 | 3,432 | 0.12444 | 0.04768 | 20.52 | 0.001 |

Thus ordinary physical line wrapping in the actual Hebrew comparator does not
produce the Voynich effect.  The Hebrew lines are automatically aligned and
selected for HTR quality, not a diplomatic layout edition, but that caveat is
unlikely to explain a null-centered result versus the enormous A/B effects.

### Hébreu 1199 transcription and image-retrieval pilots

I also tested whether the untranscribed illustrated folios could supply more
anchors.  Kraken's open [“Medieval Hebrew manuscripts in Italian
bookhand”](https://doi.org/10.5281/zenodo.5468573) recognition model is
appropriately matched to the script.  On folio 67v its
raw neural-layout transcription had a global character error rate (CER) of
30.0% against the BnF catalog.  Twenty-four automatically aligned lines were
used for a deliberately small fine-tune and seven disjoint lines on that page
were held out.  Held-out character accuracy rose from 64.7% to 81.5% (CER
35.3% to 18.5%), and exact-word accuracy from 20.4% to 61.1%.

That apparent success did **not** generalize.  On the next page, 68r, neural
layout analysis fragmented the text and scrambled reading order.  With the
same 26 independently detected and binarized horizontal line crops for both
models, the stock model's whole-page CER was 62.1% and the adapted model's was
64.9%.  The one-page adaptation had largely learned page-local hands and
segmentation conditions.  It is therefore useful training infrastructure, not
a new transcription.  A credible expansion needs manually checked PAGE-XML
from multiple folios and hands, with entire folios kept out of training.

A visual retrieval route failed its own gate as well.  BnF Latin 17844
([digital facsimile](https://gallica.bnf.fr/ark:/12148/btv1b10032359h)) is a
1440–60 northern-Italian manuscript in the direct branch of the same 98-plant
tradition, so known Hébreu 1199 correspondences provide a validation set.  A
self-supervised DINOv2 image embedding ranked the correct Latin image at 4,
97, 78, 1, 50, and 40 out of 98 for six nonblank anchors.  Only the highly
distinctive mandrake was a reliable top match.  Ordinary plant depictions vary
too much across copies for this method to justify assigning names to Voynich
drawings; no Voynich candidates were generated from the failed retrieval.

The highest-value next corpus is the full image set of Hébreu 1199, which is
available through [Gallica's IIIF manifest](https://gallica.bnf.fr/iiif/ark:/12148/btv1b10545274f/manifest.json)
but lacks a full public transcription.  A paleographically validated HTR
transcription would allow page-entry structure, plant-name captions, recipe
formulae, and line-position effects to be compared directly.  Two smaller
nomenclature controls are also promising: a [circa-1500 Latin-German
pharmaceutical glossary written in Hebrew characters](https://www.nli.org.il/he/books/NNL_ALEPH997009647486305171/NLI)
and a [fifteenth-century Italo-Romance medico-botanical glossary in Hebrew
script](https://cris.bgu.ac.il/en/publications/a-glossary-of-latin-and-italo-romance-medico-botanical-terms-in-h/).
Those are better suited to testing a borrowed plant-name layer than to training
a translation model.

## A held-out mechanism tournament

The preceding tests reject individual predictions, but they do not directly
ask which production mechanism predicts unseen manuscript material best.  I
therefore implemented an initial likelihood tournament over five folds, always
holding out complete IVTFF quires.  It covers 206 paragraph-bearing pages,
34,411 certain words, and 208,733 scored raw-EVA units (one end-of-word event is
included per token).

The meaningful-language competitor is intentionally favorable to the Hebrew
hypothesis.  A character trigram model is trained externally on 65,413 words
from Book II of the 1491 Hebrew *Canon* OCR plus the 880-word Hébreu 1199 catalog
transcript.  Another 16,655 *Canon* words are never used to train that source
model and provide a real-Hebrew calibration.  Normalized Hebrew has 22 letters
while raw EVA has 21 codepoints, so the two least frequent Hebrew letters in
the external training corpus (`צ` and `ז`) are merged into one source class.
This choice is made without Voynich data and gives a proper 21-to-21 bijection
rather than silently discarding source probability.

For every Voynich fold, simulated annealing with five restarts and 6,000 steps
per restart fits a monoalphabetic key on the training quires only, followed by a
deterministic pair-swap polish.  Forward and within-word-reversed readings
compete on training likelihood; the selected orientation and key are then
frozen for the held-out quires.  This model does not even require Hebrew syntax
or word frequencies—only within-word character behavior—so it is a weaker and
more generous target than readable Hebrew prose.

The three Voynich-native competitors use the same trigram order and smoothing:

- an isolated-word morphology model;
- that model conditioned on Currier stratum and whether the word is first,
  middle, last, or alone on its physical line;
- a proper-probability mixture of the layout model and an edit transducer that
  can copy, substitute, delete, or insert relative to the preceding word.

| Mechanism | Held-out bits/raw-EVA+EOW unit | Gain over Hebrew substitution | Fold wins |
|---|---:|---:|---:|
| Hebrew trigram + fitted substitution | 3.7344 | — | 0/5 |
| Voynich-native word morphology | 1.9281 | 1.8063 | 0/5 |
| Morphology + Currier/line position | 1.8525 | 1.8819 | 0/5 |
| Layout + previous-word copy/mutation | **1.8516** | **1.8828** | 5/5 |

The difference is large: relative to the layout-conditioned model, the Hebrew
substitution spends 392,822 additional bits on the held-out data—about 49 kB.
Every fold has the same ordering.  An order-1 sensitivity, which removes the
trigram model's higher-order advantage, still scores Hebrew substitution at
3.6057 bits/unit versus 2.0497 for the layout model, a 1.556-bit gap.

There are two nuances.  First, all five folds select within-word reversal for
the Hebrew model.  That direction is compatible with an RTL source and should
not be hidden.  It is not enough to produce a stable translation: fitted keys
agree on only 42.9% of symbol assignments on average across fold pairs.  On
real held-out Hebrew, 72.4% of length-three-or-longer words occur in the source
training lexicon; only 12.6% of the decoded held-out Voynich words do.  The
source model scores real held-out Hebrew at 3.2384 bits/unit, versus 3.7344 for
the optimally keyed Voynich.

Second, the nominal winning copy model is almost entirely its layout base.  Its
fitted copy mixture weight is only 0.30–0.38%, and it improves held-out loss by
just 0.00090 bit/unit (187 bits over the complete evaluation).  This is not
substantial evidence that the scribe repeatedly mutated the immediately
preceding word.  The robust gains are 1.806 bits/unit from learning native
Voynich morphology and another 0.0757 from supplying layout.  A broader
self-citation model may copy from a page-local pool, paradigms, or previous
lines, but that remains to be tested explicitly.

This tournament is the strongest direct comparison here, but its scope remains
finite.  It rejects **period Hebrew orthography under one stable substitution**;
it does not reject changing keys, nulls, verbose codes, nomenclators, altered
word boundaries, or a meaningful notation that is intrinsically low entropy.
Nor does it include the description length of the model families themselves.
What it establishes is that unseen Voynich is dramatically better predicted by
its own layout-coupled morphological process than by the strongest simple
Hebrew substitution tested.

## Hypothesis ledger

| Hypothesis | What it explains | Main contradiction | Current status |
|---|---|---|---|
| Ordinary plaintext in an unknown script | Frequencies, word-like morphology, sections | Very weak cross-word order; extreme line effects | Strongly disfavored under EVA word boundaries |
| Simple monoalphabetic substitution | Same as plaintext while hiding letters | Preserves sequence/repetition information; fitted Hebrew model loses about 1.81–1.88 held-out bits/unit to native/layout models | Strongly disfavored |
| Hebrew/other abjad plus word anagramming | Word-shape rank; herbal-specific match to period Hebrew controls; low vowels | No coherent output; herbal formula sequences absent; no Hebrew zodiac vocabulary under global exact keys; order signal too weak | Shape-level lead only; simple Hebrew translation/cipher versions strongly disfavored |
| Verbose, lossy, or homophonic cipher | Can suppress visible lexical/syntactic repetition | Must also explain line position and local variants; very flexible | Viable but presently underspecified |
| Code, notation, or steganographic carrier | Low linguistic order with genuine page/section structure | No recovered codebook or payload | Viable; high-value target |
| Layout-conditioned pseudotext / self-citation | Local variants, low entropy, line effects, weak long-range order; best held-out likelihood here | Immediate previous-word copying adds almost nothing; must still explain section/hand differences and image-text anchors | Best fit among tested simple mechanisms; specific generator not identified |

No purely textual statistic can prove “nonsense” against an arbitrarily
powerful cipher: such a cipher can map any plaintext to any ciphertext.  A fair
claim must compare explicit mechanisms by out-of-sample likelihood or minimum
description length, not merely note that one can imitate selected summary
statistics.

## What would constitute real progress toward decipherment

The next phase should not ask a language model to produce plausible glosses.
It should force candidate mechanisms to make risky, manuscript-wide
predictions.

1. **Reconstruct the unit inventory from images.** Fit grapheme/allograph
   segmentation jointly across multiple transcribers and, ideally, Yale page
   images.  Entropy and Hebrew-shape results must then survive posterior
   uncertainty over glyph boundaries.
2. **Extend the held-out generator tournament.** The first comparison now
   favors native layout-conditioned morphology over stable Hebrew substitution,
   but the language side still needs homophonic/verbose finite-state
   transducers, changing-key and nomenclator models.  The procedural side needs
   page-local paradigms, previous-line pools, and table/grille generators.  Use
   prequential description length as well as likelihood so added machinery is
   charged rather than rewarded for flexibility.
3. **Use illustrations only as preregistered anchors.** The zodiac audit shows
   that even obvious pictures are not necessarily direct text cribs: their
   `Lz` items label surrounding figures, not demonstrably the central signs.
   Repeated diagram labels and unambiguous numeric/calendar structure remain
   better tests than imaginative plant names.  A proposed key must be frozen
   before looking at held-out labels.
4. **Require function words and grammar.** A translation must recover frequent
   grammatical items at plausible rates, preserve their distribution across
   word lengths and positions, and yield coherent unseen passages without
   spelling repair or anagram cherry-picking.
5. **Measure payload.** If a code/steganographic model is proposed, estimate the
   recoverable bits per line and show a stable decoder on held-out pages.  If a
   pseudotext model is proposed, show that it compresses the manuscript better
   than the best meaningful-code alternative.

The low-hanging fruit was not a hidden English sentence waiting for a larger
Transformer.  It was a set of methodological cleanups and falsification tests:
correct boundaries, representation sensitivity, cross-word rather than raw
loss, independent transcription, and adversarial language-ID controls.  These
move the problem away from ordinary substitution and toward the manuscript's
actual hard question: **what production mechanism creates word-like local
forms, weak sentence-like order, and extraordinarily strong layout coupling?**

## Reproduction

The primary audit is dependency-free:

```bash
python3 scripts/audit_voynich.py \
  --permutations 499 \
  --control English=data/capote_char/clean_incoldblood.txt \
  --control French=data/aurebours_char/clean_aurebours.txt \
  --control Tagalog=data/noli_me_tangere_char/clean_noli_me_tangere.txt \
  --control Hawaiian=data/hawaiian_char/clean_hawaiian.txt
```

The language-hypothesis audit uses the public NLTK UDHR data:

```bash
curl -L https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip -o /tmp/udhr.zip
unzip /tmp/udhr.zip -d /tmp/udhr
python3 scripts/audit_language_hypotheses.py \
  --udhr-dir /tmp/udhr/udhr \
  --bootstrap-iterations 200 \
  --json
```

The period Hebrew medical comparison uses the OCR XML for the 1491 *Canon*
scan and optionally the BnF catalog's human-transcribed herbal sample:

```bash
curl -L \
  https://archive.org/download/4072969.med.yale.edu/4072969.med.yale.edu_djvu.xml \
  -o /tmp/avicenna_djvu.xml
curl -L -A 'Mozilla/5.0' \
  https://archivesetmanuscrits.bnf.fr/ark:/12148/cc8082r \
  -o /tmp/hebreu1199_bnf_catalog.html
python3 scripts/compare_medical_hebrew.py \
  --canon-xml /tmp/avicenna_djvu.xml \
  --direct-herbal-catalog-html /tmp/hebreu1199_bnf_catalog.html \
  --bootstrap-iterations 200
```

The formula-sequence test uses the same downloaded BnF catalog page:

```bash
python3 scripts/compare_hebrew_herbal_structure.py \
  --catalog-html /tmp/hebreu1199_bnf_catalog.html \
  --bootstrap-iterations 1000
```

The zodiac anchor test is dependency-free.  Its candidate vocabulary,
selection rule, folds, and permutation unit are declared in the script:

```bash
python3 scripts/zodiac_hebrew_anchors.py \
  --permutations 499 \
  --beam-size 50000
```

The mechanism tournament uses the already downloaded period Hebrew controls:

```bash
python3 scripts/compare_mechanisms.py \
  --canon-xml /tmp/avicenna_djvu.xml \
  --catalog-html /tmp/hebreu1199_bnf_catalog.html \
  --folds 5 \
  --order 2 \
  --key-restarts 5 \
  --key-steps 6000
```

The HTR preparation tool takes a Kraken PAGE-XML file containing baseline OCR
and transfers line boundaries from the catalog alignment while keeping the
training and evaluation lines disjoint:

```bash
python3 scripts/prepare_hebrew_htr_adaptation.py \
  --page-xml /tmp/kraken_67v_neural.xml \
  --catalog-html /tmp/hebreu1199_bnf_catalog.html \
  --train-xml /tmp/hebreu1199_htr/train.xml \
  --evaluation-xml /tmp/hebreu1199_htr/eval.xml \
  --all-xml /tmp/hebreu1199_htr/all.xml \
  --folio 67v

python3 scripts/compare_hebrew_herbal_structure.py \
  --catalog-html /tmp/hebreu1199_bnf_catalog.html \
  --bootstrap-iterations 1000 \
  --hebrew-page-xml /tmp/hebreu1199_htr/train.xml \
  --hebrew-page-xml /tmp/hebreu1199_htr/eval.xml \
  --line-permutations 999
```

Run the regression suite with:

```bash
python3 -m unittest discover -s tests -v
```

### Caution on a current preprint

I also inspected the code accompanying the 2026 preprint [*A Statistical Turing
Test for the Voynich Manuscript*](https://www.researchsquare.com/article/rs-9755825/latest.pdf).
It is a useful catalog of 18 generator families, but two implementation choices
make its pass counts unsuitable as decisive evidence.  Its M8 code accepts
`voynich_accuracy - synthetic_accuracy < 0.05`, without an absolute value;
models far *more* classifiable than Voynich therefore pass (reported deltas
reach −0.370).  Its M7 baseline uses named Voynich sections while synthetic
texts are divided into five positional chunks; the repository itself records a
different result when Voynich is split positionally.  These issues do not
invalidate the generated corpora, but they do invalidate a literal reading of
the eight-test pass total.
