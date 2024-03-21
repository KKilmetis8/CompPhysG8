% See: http://www.dante.de/help/latex/guide/clsguide/clsguide.html
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IAYCPro}[2014/09/04 v0.4 Basic class for IAYC reports]

% Main document class
\LoadClass[12pt,a4paper,twoside, parskip=half]{scrartcl}

% Packages

% AMS Packages for mathematical stuff
\RequirePackage{amsmath,amsfonts,amssymb,amsthm}

% Header styling
\RequirePackage{fancyhdr}

% Input and output encoding options, document language -> English
\RequirePackage[latin1]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}

% Graphics packages
\RequirePackage{float}
\RequirePackage{graphicx}

% Verbatim for displaying code in reports
\RequirePackage{verbatim}

% Page styling
\pagestyle{fancy}

\renewcommand{\headrulewidth}{0.4pt}
\setlength{\topmargin}{-1.54cm}
\setlength{\headheight}{0.6cm}
\setlength{\headsep}{1cm}
\setlength{\textheight}{24.3cm}
\setlength{\footskip}{1.2cm}
\setlength{\oddsidemargin}{-0.54cm}
\setlength{\evensidemargin}{-0.54cm}
\setlength{\textwidth}{17cm}
\setlength{\headwidth}{17cm}

% Title spacing
% \RequirePackage{titlesec}
% \titlespacing{\section}{0pt}{10pt}{10pt}

% Adjusted float rules
\renewcommand{\floatpagefraction}{.6} %  Default: .5 
\renewcommand{\textfraction}{.15}  % Default: .2

% Custom IAYC title section
\renewcommand{\maketitle}{
  \begin{center}
    {\LARGE \textsc{ \@projectname\\[0.3cm]}}
    {\Large
      \@projectauthor
    }
  \end{center}
}

\newcommand{\projectname}[1]{\renewcommand{\@projectname}{#1}}
\newcommand{\@projectname}{\ClassError{IAYCPro}
  {You have not specified a proper project name}
  {Solution: Use \protect\projectname{123}} }

\newcommand{\projectauthor}[1]{\renewcommand{\@projectauthor}{#1}}
\newcommand{\@projectauthor}{\ClassError{IAYCPro}
  {You have not specified a particular author. }
  {Solution: Use \protect\projectauthor{I am the King}}}

\newcommand{\projectyear}[1]{\renewcommand{\@projectyear}{#1}}
\newcommand{\@projectyear}{\ClassError{IAYCPro}
  {You have not specified a year. }
  {Solution: Use \protect\projectyear{1920}}}

\newcommand{\projectgroup}[1]{\renewcommand{\@projectgroup}{#1}}
\newcommand{\@projectgroup}{\ClassError{IAYCPro}
  {You have not specified a working group. }
  {Solution: Use \protect\projectgroup{ABC}}}

\newcommand{\projectstartpage}[1]{\renewcommand{\@projectstartpage}{#1}}
\newcommand{\@projectstartpage}{\ClassError{IAYCPro}
  {Please specify a starting page.}
  {Solution: Use \protect\projectstartpage{42}}}

% IAYC header, footer and maketitle
\AtBeginDocument{
  \setcounter{page}{\@projectstartpage}
  \lhead[ - \arabic{page} -]{\normalfont \scshape \@projectname}
  \rhead[\normalfont \scshape \@projectgroup]{ - \arabic{page} -}
  \cfoot{}
  \maketitle
}

% Providing the Theorem-like structures: Theorem, Example, Definition
    \theoremstyle{definition}
    \newtheorem{definition}{Definition}[section]
    
    \theoremstyle{remark}
    \newtheorem{example}{Example}[section]
    \newtheorem*{remark}{Remark}
    
    \theoremstyle{plain}
    \newtheorem{theorem}{Theorem}[section]
    \newtheorem*{assertion}{Assertion}
    
% Eq.number depends on the section: eq. (7.324)
    \numberwithin{equation}{section}