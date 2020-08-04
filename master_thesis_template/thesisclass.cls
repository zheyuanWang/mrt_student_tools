%% LaTeX2e class for diploma theses
%% Based on wissdoc.cls by Roland Bless, 1996-2001
%% bless@telematik.informatik.uni-karlsruhe.de
%%
%% Adapted by:
%%  - Timo Rohrberg, 2009, timo.rohrberg@student.kit.edu
%%  - Maximilian Naumann, 2016, maximilian.naumann@student.kit.edu


\NeedsTeXFormat{LaTeX2e}										% We do need LaTeX2e
\ProvidesClass{thesisclass}
%\LoadClass[a4paper,11pt,titlepage, oneside]{scrbook}	% onesided
\LoadClass[a4paper,11pt,titlepage]{scrbook} 			% twosided

\ProcessOptions\relax


%% -------------------------------
%% |          Packages           |
%% -------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}   % Input in UTF8
\RequirePackage{ae}               % Almost european, virtual T1-Font
\RequirePackage[pdftex]{graphicx}
\RequirePackage{vmargin}          % Adjust margins in a simple way
\RequirePackage{fancyhdr}         % Define simple headings
\RequirePackage{subfigure}
\RequirePackage{url}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{tikz}
%\RequirePackage[english,ngerman]{babel}		% English and German
\RequirePackage{mathtools} 				% für die benutzung ausgefeilter formel-formatierungen
\RequirePackage{placeins}					% um das darstellen von float-umgebungen zu erzwingen
\RequirePackage{tabularx} 				% für umgebrochene tabellen
\RequirePackage{color}						% erlaubt das Benutzen von Farben
\RequirePackage[raiselinks=true,
						bookmarks=true,
						bookmarksopenlevel=1,
						bookmarksopen=true,
						bookmarksnumbered=true,
						hyperindex=true,
						plainpages=false,
						pdfpagelabels=true,
						linkcolor=black,
						citecolor=black, %keiner will bunte Links ...
						pdfborderstyle={/S/U/W 1},
						pdfborder={0 0 0.5}]{hyperref}
\RequirePackage{blindtext}

%% --- End of Packages ---



%% -------------------------------
%% |        Declarations         |
%% -------------------------------
\DeclareGraphicsExtensions{.svg, .pdf, .png}
\graphicspath{{Graphics/}}     % Bilder werden im "Graphics"-Ordner gesucht
%% --- End of Declarations ---



%% -------------------------------
%% |         New commands        |
%% -------------------------------
\newcommand{\changefont}[3]{\fontfamily{#1} \fontseries{#2} \fontshape{#3} \selectfont}
%\renewcommand*{\headfont}{\slshape}
%\newcommand{\captionfont}{}
\newcommand{\chapterheadfont}{}
\newcommand{\blankpage}{
 \clearpage{\pagestyle{empty}\cleardoublepage}
}
%% --- End of New Commands ---



%% -------------------------------
%% |      Globale Settings       |
%% -------------------------------
\setcounter{secnumdepth}{3} % Numbering also for \subsubsections
\setcounter{tocdepth}{2}    % dont Register \subsubsections in content directory

\setpapersize{A4}
\setmarginsrb{3cm}{1cm}{3cm}{1cm}{6mm}{7mm}{5mm}{15mm}

\parindent 0cm                     % Do not indent beginning of paragraph
\parskip1.5ex plus0.5ex minus0.5ex % Margin between paragraphs
%% --- End of global Settings ---



%% -------------------------------
%% |          Headings           |
%% -------------------------------
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\fancyhf{}
\fancyhead[LE,RO]{{\headfont\thepage}}						% Left/right header for even/odd pages
\fancyhead[LO]{\headfont\nouppercase{\rightmark}}	% Header for left page (odd)
\fancyhead[RE]{\headfont\nouppercase{\leftmark}}	% Header for right page (even)
%\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{%
\fancyhf{}													% No Header and Footer fields
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
%\fancyfoot[C]{\thepage}
}
%% --- End of Headings ---



%% -------------------------------
%% |      Style of captions      |
%% -------------------------------
\renewcommand{\chaptername}{}

\renewcommand{\section}{%
\@startsection{section}%
{1}%		% Structure level
{0mm}%	% Indention
{1.5ex plus 1ex minus 1.2ex}%			% Pre-Margin
{0.5ex plus 0.5ex minus 0.5ex}%		% Post-Margin
{\chapterheadfont\Large\bfseries}%	% Style
}
\renewcommand{\subsection}{%
\@startsection{subsection}%
{2}		% Structure level
{0mm}	% Indention
{1ex plus 1ex minus 1ex}					% Pre-Margin
{0.3ex plus 0.3ex minus 0.3ex}		% Post-Margin
{\chapterheadfont\large\bfseries}	% Style
}
\renewcommand{\subsubsection}{%
\@startsection{subsubsection}%
{3}		% Structure level
{0mm}	% Indention
{1ex plus 1ex minus 1ex}								% Pre-Margin
{0.2ex plus 0.2ex minus 0.2ex}					% Post-Margin
{\chapterheadfont\normalsize\bfseries}	% Style
}
\renewcommand{\paragraph}{%
\@startsection{paragraph}%
{4}		% Structure level
{0mm}	% Indention
{2ex plus 1ex minus 2ex}								% Pre-Margin
{0.2ex plus 0.2ex minus 0.2ex}					% Post-Margin
{\chapterheadfont\normalsize\bfseries}	% Style
}
%% --- End of captions style ---



%% ---------------------------------
%% |   Style of chapter captions   |
%% ---------------------------------
\newlength{\chapnolen}
\newlength{\chapparlen}
\newsavebox{\chapno}
\renewcommand{\@makechapterhead}[1]{
  \vspace*{0.2\textheight}
  \vskip 15\p@
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \savebox{\chapno}{\chapterheadfont\huge\bfseries \thechapter.}
        \settowidth{\chapnolen}{\usebox{\chapno}}
        \parbox[t]{\chapnolen}{\usebox{\chapno}}\nobreak\leavevmode
      \fi
    \fi
    \interlinepenalty\@MM
    \setlength{\chapparlen}{\textwidth}
    \addtolength{\chapparlen}{-1.0\chapnolen}
    \addtolength{\chapparlen}{-2ex}
    \leavevmode\nobreak
    \parbox[t]{\chapparlen}{\raggedright\chapterheadfont\huge \bfseries #1\par\nobreak}
    \vskip 40\p@
  }}

\renewcommand{\@makeschapterhead}[1]{
  \vspace*{50\p@}
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \chapterheadfont \huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }
}
%% --- End of chapter captions style ---

%% -------------------------------
%% |			  Nomenclature				 |
%% -------------------------------
%\let\sym\nomenclature
%\renewcommand{\nomname}{Acronyms}
%\newcommand{\nomunit}[1]{\renewcommand{\nomentryend}{\hspace*{\fill}#1}}
%\setlength{\nomlabelwidth}{.20\hsize}
%\setlength{\nomitemsep}{-\parsep}
%\makenomenclature
%\sym[00]{}{} %hier kann man die erste Zeile der Tabelle festlegen
%\sym[01]{}{}

%% -------------------------------
%% |  Style of content directory |
%% -------------------------------
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{{\pdfbookmark{\contentsname}{\contentsname}\chapterheadfont\oldtableofcontents}}
\let\@olddottedtocline\@dottedtocline
\renewcommand{\@dottedtocline}[5]{\@olddottedtocline{#1}{#2}{#3}{#4}{\chapterheadfont #5}}
%% --- End of content directory style ---

%% **** END OF CLASS ****
