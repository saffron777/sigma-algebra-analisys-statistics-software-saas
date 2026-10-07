"""Static content of the Help page: user guide and the theoretical reference of every analysis module.

The reference is written at the level of a mathematical statistician: formal definitions, key
formulas (LaTeX, rendered client-side by KaTeX) and illustrative SVG figures. Kept as light HTML.
"""

from __future__ import annotations

from sigma.presentation.web import help_figures

# --------------------------------------------------------------------------- helpers
def _fig(key: str, caption: str) -> str:
    svg = help_figures.FIGURES[key]()
    return (
        f'<figure class="my-3 overflow-x-auto rounded-sm border border-slate-200 p-2 dark:border-slate-700">{svg}'
        f'<figcaption class="mt-1 text-[11px] text-slate-500 dark:text-slate-400">{caption}</figcaption></figure>'
    )


# --------------------------------------------------------------------------- user guide
GUIDE: list[tuple[str, str]] = [
    (
        "Importar datos",
        """
        <p>Sigma guarda cada dataset en su propia base de datos local. Para empezar, vaya a
        <strong>Datos → Importar</strong> (o el icono de subir de la barra de herramientas):</p>
        <ul>
          <li><strong>Excel, CSV o texto</strong>: elija el archivo, el delimitador, la codificación y si la
          primera fila son encabezados, y pulse <strong>Vista previa</strong>.</li>
          <li><strong>Bases de datos</strong> (SQL Server, MySQL/MariaDB, PostgreSQL, Oracle, MongoDB):
          complete la conexión, elija tabla o consulta y confirme la vista previa.</li>
          <li><strong>API JSON</strong>: indique la URL, cabeceras y parámetros; Sigma aplana los objetos en columnas.</li>
        </ul>
        <p>Antes de guardar puede ajustar el tipo de cada columna (numérica, categórica, texto, fecha, lógica).
        El dataset activo se muestra en la barra de estado inferior.</p>
        """,
    ),
    (
        "Ver y preparar los datos",
        """
        <p>Abra un dataset con <strong>Ver</strong> para inspeccionarlo:</p>
        <ul>
          <li><strong>Vista de datos</strong>: las filas (casos) y columnas (variables), con paginación.</li>
          <li><strong>Vista de variables</strong>: una fila por variable, donde puede cambiar su tipo.</li>
        </ul>
        <p>Para limpiar, calcular o unir columnas use <strong>Datos → Transformar</strong> (recodificar, filtrar,
        imputar valores faltantes, estandarizar, crear variables con fórmulas…).</p>
        """,
    ),
    (
        "Ejecutar un análisis",
        """
        <p>Desde el menú <strong>Analizar</strong> (o <strong>Gráficos</strong>) elija el procedimiento.
        Cada página pide <strong>roles</strong> (qué columnas cumplen cada papel) y <strong>parámetros</strong>
        (nivel de significancia, modelo, alternativas…), y muestra un <strong>Ver código</strong> con el equivalente
        en Python (<code>sigma.load(...)</code>) para reproducirlo en la consola.</p>
        """,
    ),
    (
        "Guardar, exportar e imprimir",
        """
        <ul>
          <li><strong>Guardar resultado</strong> lo conserva en la base de datos local (visor de dos paneles en
          <strong>Archivo → Resultados guardados</strong>).</li>
          <li><strong>Exportar</strong> genera un informe en PDF, Word o Excel; puede combinar varios resultados
          en un único documento con portada e índice.</li>
          <li><strong>Imprimir</strong> prepara tablas y gráficos para papel o PDF.</li>
        </ul>
        """,
    ),
    (
        "Consola (Python y R) y asistente",
        """
        <p>La <strong>consola</strong> abre notebooks Jupyter en Python o en R; en ambos, <code>sigma</code> da
        acceso a sus datos (<code>sigma$load("nombre")</code>, <code>sigma$save(df, "nombre")</code>).</p>
        <p>El <strong>asistente</strong> le recomienda el procedimiento adecuado a partir de su objetivo y del
        tipo de variables.</p>
        """,
    ),
]

# --------------------------------------------------------------------------- reference
REFERENCE: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Estadística descriptiva",
        [
            (
                "Estadística descriptiva",
                "descriptiva",
                """
                <p>Sea una muestra $x_1, \\dots, x_n$ de una variable numérica. La descriptiva condensa su
                <em>centro</em>, su <em>dispersión</em> y su <em>forma</em>.</p>

                <p><strong>Tendencia central.</strong> La media muestral es el momento de primer orden,
                $$ \\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i, $$
                que minimiza la suma de desviaciones al cuadrado, $\\sum (x_i - c)^2$. La <em>mediana</em> es el
                valor que deja la mitad de las observaciones a cada lado (el cuantil 0.5); la <em>moda</em> es el
                valor más frecuente. Con datos asimétricos la media se desplaza hacia la cola larga mientras la
                mediana permanece robusta.</p>

                <p><strong>Dispersión.</strong> La varianza muestral (estimador insesgado) y la desviación estándar,
                $$ s^2 = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2, \\qquad s = \\sqrt{s^2}, $$
                miden la dispersión media en torno a la media. El rango intercuartílico $\\mathrm{IQR} = Q_3 - Q_1$
                es robusto a atípicos; el coeficiente de variación $\\mathrm{CV} = s/\\bar{x}$ expresa la dispersión
                en términos relativos.</p>

                <p><strong>Forma.</strong> La <em>asimetría</em> (momento estandarizado de tercer orden),
                $$ g_1 = \\frac{1}{n}\\sum_{i=1}^{n}\\left(\\frac{x_i - \\bar{x}}{s}\\right)^3, $$
                vale $0$ en distribuciones simétricas, es positiva cuando la cola derecha es más larga y negativa
                en caso contrario. La <em>curtosis</em> (momento de cuarto orden, restándole $3$) mide el
                apuntamiento: valores positivos indican colas más pesadas que la normal.</p>

                <p><strong>Interpretación.</strong> Con distribuciones aproximadamente simétricas reporte
                $\\bar{x} \\pm s$; con asimetría prefiera mediana e IQR. Siempre acompañe con un histograma o un
                diagrama de caja y revise los valores faltantes y atípicos antes de concluir.</p>
                """,
            ),
            (
                "Distribuciones de probabilidad",
                "distribuciones",
                """
                <p>Una <strong>variable aleatoria</strong> $X$ queda descrita por su función de distribución
                acumulada $F(x) = P(X \\le x)$. Si $X$ es continua, su <em>densidad</em> $f(x) = F'(x)$ satisface
                $\\int f = 1$ y $P(a \\le X \\le b) = \\int_a^b f(x)\\,dx$; si es discreta, su función de masa
                $p(k) = P(X = k)$ satisface $\\sum_k p(k) = 1$.</p>

                <p>La <strong>normal</strong> es el límite central de la inferencia: si
                $X_1,\\dots,X_n \\stackrel{\\mathrm{iid}}{\\sim} (\\mu, \\sigma^2)$, entonces
                $\\sqrt{n}(\\bar{X}_n - \\mu)/\\sigma \\xrightarrow{d} N(0,1)$. Su densidad es
                $$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}, \\qquad X \\sim N(\\mu, \\sigma^2). $$
                La <strong>t de Student</strong>, $T \\sim t_\\nu$, surge al estandarizar una media con $\\sigma$
                desconocida; tiene colas más pesadas que la normal y converge a ella cuando $\\nu \\to \\infty$.
                La <strong>chi-cuadrado</strong>, $\\chi^2_\\nu = \\sum_{i=1}^{\\nu} Z_i^2$ con $Z_i \\sim N(0,1)$,
                y la <strong>F de Fisher</strong>, $F = (\\chi^2_{a}/a)/(\\chi^2_{b}/b)$, sustentan las pruebas de
                varianza, ANOVA y la significancia global de la regresión.</p>

                <p><strong>Conteos.</strong> La <strong>binomial</strong> $X \\sim \\mathrm{Bin}(n, p)$ cuenta éxitos
                en $n$ ensayos independientes, $P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k}$, con $E[X]=np$ y
                $\\mathrm{Var}(X)=np(1-p)$. La <strong>Poisson</strong> $X \\sim \\mathrm{Pois}(\\lambda)$ modela
                eventos raros, $P(X=k) = e^{-\\lambda}\\lambda^k/k!$, con $E[X]=\\mathrm{Var}(X)=\\lambda$; es el
                límite de la binomial cuando $n\\to\\infty$, $p\\to 0$ con $np = \\lambda$. La
                <strong>geométrica</strong> cuenta ensayos hasta el primer éxito; la <strong>hipergeométrica</strong>
                modela extracciones sin reemplazo; la <strong>binomial negativa</strong> cuenta fracasos antes de
                $r$ éxitos y es la alternativa estándar a Poisson cuando hay <em>sobredispersión</em>.</p>

                <p><strong>Distribuciones de conteo (discretas).</strong> Funciones de masa de probabilidad
                $P(X=k)$ de las cinco distribuciones discretas del módulo:</p>
                """ + _fig("binomial", "Binomial(n=10, p=0.5): éxitos en 10 ensayos independientes; E = np, Var = np(1−p).")
                + _fig("poisson", "Poisson(λ=3): conteo de eventos raros; media = varianza = λ.")
                + _fig("geometric", "Geométrica(p=0.3): ensayos hasta el primer éxito; sin memoria discreta.")
                + _fig("hypergeometric", "Hipergeométrica(M=50, K=10, n=10): extracciones sin reemplazo.")
                + _fig("negative_binomial", "Binomial negativa(r=3, p=0.5): fracasos antes de r éxitos; capta sobredispersión.")
                + """

                <p><strong>Tiempos de vida y positivas.</strong> La <strong>exponencial</strong>,
                $f(x)=\\lambda e^{-\\lambda x}$, es el tiempo entre eventos de un proceso de Poisson y es la única
                distribución continua <em>sin memoria</em>: $P(X > s+t \\mid X > s) = P(X > t)$. La <strong>gamma</strong>,
                $f(x) = x^{k-1}e^{-x/\\theta}/(\\Gamma(k)\\theta^k)$, generaliza a la suma de $k$ exponenciales. La
                <strong>Weibull</strong> modela tasas de fallo crecientes o decrecientes; la <strong>log-normal</strong>
                (logaritmo normal) describe variables positivas asimétricas como ingresos; la <strong>beta</strong>,
                $f(x) \\propto x^{\\alpha-1}(1-x)^{\\beta-1}$ sobre $[0,1]$, modela proporciones; la
                <strong>uniforme</strong> asigna densidad constante en $[a,b]$.</p>

                <p><strong>Distribuciones de referencia.</strong> Las cuatro distribuciones que sustentan la inferencia
                clásica y sus formas típicas:</p>
                """ + _fig("normal", "Normal: campana simétrica centrada en μ; el área entre μ ± σ es ≈ 68 %.")
                + _fig("t", "t de Student: colas más pesadas que la normal para muestras pequeñas; converge a N(0,1) cuando ν → ∞.")
                + _fig("chi2", "χ²: asimétrica a la derecha, soporte positivo; media = ν grados de libertad.")
                + _fig("f", "F de Fisher: cociente de dos χ² independientes escaladas; soporte positivo.")
                + """
                <p>El módulo también calcula densidad/probabilidad, acumulada, cuantiles y valores p.</p>
                """,
            ),
        ],
    ),
    (
        "Pruebas de hipótesis",
        [
            (
                "Pruebas de hipótesis",
                "hipotesis",
                """
                <p>Se contrasta una hipótesis nula $H_0$ contra una alternativa $H_1$. Un estadístico de prueba
                $T$ resume la evidencia y su distribución bajo $H_0$ permite calcular el <strong>valor p</strong>:
                $$ p = P_{H_0}(T \\text{ tan extremo como el observado}). $$
                Se rechaza $H_0$ si $p < \\alpha$ (nivel de significancia, típicamente $0.05$).</p>

                <p><strong>Errores.</strong> El <em>error tipo I</em> es $\\alpha = P(\\text{rechazar } H_0 \\mid H_0)$;
                el <em>error tipo II</em> es $\\beta = P(\\text{no rechazar } H_0 \\mid H_1)$. La
                <strong>potencia</strong> es $1 - \\beta$ y crece con el tamaño del efecto y con $n$.</p>

                <p><strong>Pruebas sobre medias.</strong> Con $\\sigma$ conocida,
                $Z = (\\bar{x}-\\mu_0)/(\\sigma/\\sqrt{n}) \\sim N(0,1)$. Con $\\sigma$ desconocida,
                $T = (\\bar{x}-\\mu_0)/(s/\\sqrt{n}) \\sim t_{n-1}$. Para dos muestras independientes
                $\\bar{x}_1 - \\bar{x}_2$ se estandariza con el error estándar agrupado (varianzas iguales) o con la
                aproximación de Welch (varianzas distintas); para muestras <em>pareadas</em> se analizan las
                diferencias $d_i = x_{1i} - x_{2i}$.</p>

                <p><strong>Proporciones.</strong> $Z = (\\hat{p} - p_0)/\\sqrt{p_0(1-p_0)/n}$, y para dos
                proporciones con la proporción agrupada. <strong>χ² de independencia</strong>:
                $\\chi^2 = \\sum (O_{ij} - E_{ij})^2 / E_{ij}$ con $E_{ij} = (\\text{fila}_i \\times
                \\text{columna}_j)/n$. <strong>F de varianzas</strong>: $F = s_1^2 / s_2^2 \\sim F_{n_1-1, n_2-1}$.
                Pruebas no paramétricas (Wilcoxon, Mann-Whitney, Kruskal-Wallis) reemplazan las anteriores cuando
                falla la normalidad.</p>

                <p><strong>Interpretación.</strong> Un $p$ pequeño no mide la <em>magnitud</em> del efecto: acompañe
                con un tamaño de efecto (d de Cohen, $r$, $\\eta^2$) y un intervalo de confianza, y verifique los
                supuestos de cada prueba. Un resultado no significativo no «prueba» $H_0$, solo que la evidencia es
                insuficiente.</p>
                """,
            ),
            (
                "Pruebas adicionales",
                "pruebas",
                """
                <p>Contrastes de supuestos y situaciones especiales:</p>
                <ul>
                  <li><strong>Normalidad</strong>: Shapiro-Wilk (basado en la regresión de los cuantiles sobre los
                  valores esperados), Kolmogorov-Smirnov ($D = \\sup_x |F_n(x) - F_0(x)|$), Anderson-Darling y
                  Jarque-Bera ($JB = n(\\hat{g}_1^2/6 + (\\hat{g}_2-3)^2/24)$).</li>
                  <li><strong>Homogeneidad de varianzas</strong>: Levene (basado en desviaciones absolutas),
                  Bartlett ($M/C \\sim \\chi^2_{k-1}$) y Brown-Forsythe (robusto).</li>
                  <li><strong>No paramétricas</strong>: Mann-Whitney/Wilcoxon (rangos), Kruskal-Wallis,
                  Friedman, signos y rachas.</li>
                  <li><strong>Otras</strong>: tasas de Poisson, equivalencia (TOST), acuerdo entre evaluadores
                  (kappa de Cohen), odds ratio, Mantel-Haenszel, T² de Hotelling y correlación parcial.</li>
                </ul>
                <p>La <strong>equivalencia</strong> invierte la lógica: se rechaza «inequivalencia» para concluir
                que dos tratamientos difieren menos que un margen $\\Delta$ preestablecido. El <strong>kappa</strong>
                corrige el acuerdo por azar: $\\kappa = (p_o - p_e)/(1 - p_e)$.</p>
                """,
            ),
            (
                "Potencia y tamaño de muestra",
                "potencia",
                """
                <p>La <strong>potencia</strong> de una prueba es $1 - \\beta = P(\\text{rechazar } H_0 \\mid H_1)$.
                Para la prueba t de una media (alternativa bilateral),
                $$ 1 - \\beta = P\\left( |T| > t_{n-1, 1-\\alpha/2} \\right), \\quad T \\sim t_{n-1}(\\delta\\sqrt{n}), $$
                donde $\\delta = (\\mu_1 - \\mu_0)/\\sigma$ es el tamaño del efecto estandarizado. Despejando $n$:
                $$ n = \\left(\\frac{z_{1-\\alpha/2} + z_{1-\\beta}}{\\delta}\\right)^2. $$
                La potencia crece con $\\delta$, con $n$ y con $\\alpha$; decrece con la variabilidad.</p>
                <p>Se calcula para pruebas t (una/dos muestras, pareadas), proporciones y ANOVA. Úselo
                <em>antes</em> de recoger datos para fijar un $n$ que garantice una potencia objetivo
                (típicamente $0.80$) frente a un efecto que sea prácticamente relevante.</p>
                """ + _fig("potencia", "Potencia en función del tamaño del efecto: crece hacia 1 al alejarse de H₀."),
            ),
            (
                "Remuestreo",
                "remuestreo",
                """
                <p>Estima la variabilidad de un estadístico $\\hat{\\theta}$ <em>sin suponer una distribución</em>,
                simulando la distribución muestral.</p>
                <p><strong>Bootstrap.</strong> Se extraen $B$ muestras con reemplazo $x^*_1, \\dots, x^*_B$ de los
                datos y se calcula $\\hat{\\theta}^*_b$ en cada una; el intervalo percentil es
                $$ \\left[\\hat{\\theta}^*_{(\\alpha/2)},\\; \\hat{\\theta}^*_{(1-\\alpha/2)}\\right], $$
                donde $\\hat{\\theta}^*_{(q)}$ es el cuantil empírico de las réplicas.</p>
                <p><strong>Permutación.</strong> Bajo $H_0$ las etiquetas de los grupos son intercambiables; se
                reordenan aleatoriamente y el valor p es la fracción de permutaciones cuyo estadístico supera al
                observado.</p>
                <p><strong>Monte Carlo.</strong> Simula directamente de una distribución elegida. Use una semilla
                fija para reproducibilidad y un $B$ suficiente (varios miles) para estabilizar los cuantiles.</p>
                """,
            ),
        ],
    ),
    (
        "Mercados y encuestas",
        [
            (
                "Análisis de mercados (muestreo)",
                "mercados",
                """
                <p>Para estimar una proporción $p$ con margen de error $e$ y confianza $1-\\alpha$, el tamaño de
                muestra (población infinita) es
                $$ n = \\frac{z_{1-\\alpha/2}^2 \\, p(1-p)}{e^2}, $$
                máximo en $p = 1/2$ (peor caso). Con población finita $N$ se aplica la corrección
                $n_c = n / (1 + (n-1)/N)$. El <strong>margen de error</strong> es la mitad de la amplitud del
                intervalo: $e = z_{1-\\alpha/2}\\sqrt{\\hat{p}(1-\\hat{p})/n}$.</p>
                <p>El <strong>muestreo estratificado</strong> asigna $n_h$ a cada estrato (proporcional u óptimo,
                $n_h \\propto N_h \\sigma_h$) y reduce la varianza cuando los estratos son homogéneos internamente;
                el <strong>por conglomerados</strong> es más barato pero menos eficiente (efecto de diseño
                $\\mathrm{deff} = 1 + (m-1)\\rho$).</p>
                """,
            ),
            (
                "Regresión de encuestas",
                "encuestas",
                """
                <p>Con pesos muestrales $w_i$, los estimadores ponderan cada observación según su probabilidad de
                selección $\\pi_i$ (peso $w_i = 1/\\pi_i$). La media ponderada es
                $$ \\bar{y}_w = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}, $$
                y la regresión pondera la pseudo-verosimilitud. Los errores estándar se corrigen por el diseño
                (estratos y conglomerados) mediante linealización (Taylor) o replicación. El <em>efecto de diseño</em>
                $\\mathrm{deff} = \\mathrm{Var}_{\\text{diseño}}/\\mathrm{Var}_{\\text{MAS}}$ mide cuánto pierde de
                eficiencia el diseño frente al muestreo aleatorio simple.</p>
                <p>Use este módulo con muestras complejas: ignorar los pesos sesga las estimaciones hacia los grupos
                sobrerrepresentados.</p>
                """,
            ),
        ],
    ),
    (
        "Regresión y modelos",
        [
            (
                "Regresión lineal",
                "regresion",
                """
                <p>El modelo es $y_i = x_i^\\top \\beta + \\varepsilon_i$, con $\\varepsilon_i \\stackrel{\\mathrm{iid}}{\\sim}
                (0, \\sigma^2)$. En notación matricial $y = X\\beta + \\varepsilon$; el estimador de
                <strong>mínimos cuadrados ordinarios</strong> minimiza $\\|y - X\\beta\\|^2$ y vale
                $$ \\hat{\\beta} = (X^\\top X)^{-1} X^\\top y, \\qquad \\mathrm{Var}(\\hat{\\beta}) = \\sigma^2 (X^\\top X)^{-1}. $$
                Los residuos son $e = y - X\\hat{\\beta}$ y el estimador insesgado de la varianza,
                $\\hat{\\sigma}^2 = e^\\top e / (n - p - 1)$. El coeficiente de determinación
                $$ R^2 = 1 - \\frac{\\sum e_i^2}{\\sum (y_i - \\bar{y})^2} $$
                es la proporción de varianza explicada.</p>
                <p><strong>Inferencia.</strong> Bajo normalidad de los errores,
                $\\hat{\\beta}_j \\sim N(\\beta_j, \\sigma^2 [(X^\\top X)^{-1}]_{jj})$, de donde
                $t = \\hat{\\beta}_j / \\mathrm{se}(\\hat{\\beta}_j) \\sim t_{n-p-1}$. La significancia global usa
                $F = (R^2/p)/((1-R^2)/(n-p-1)) \\sim F_{p, n-p-1}$.</p>
                <p><strong>Supuestos.</strong> Linealidad, independencia, homocedasticidad, normalidad de los
                errores y ausencia de multicolinealidad perfecta. Diagnostique con los residuos (gráficos y pruebas)
                y use la predicción con su intervalo de confianza y de predicción.</p>
                """ + _fig("regresion", "Recta de mínimos cuadrados y un residuo εᵢ = yᵢ − ŷᵢ."),
            ),
            (
                "Modelos de regresión (logística, Poisson, GLM…)",
                "modelos",
                """
                <p>Un <strong>modelo lineal generalizado (GLM)</strong> generaliza la regresión a respuestas no
                normales mediante una <em>liga</em> $g$: $g(\\mu_i) = x_i^\\top \\beta$, con $\\mu_i = E[y_i]$.</p>
                <p><strong>Logística</strong> (binomial, liga logit): $\\log\\frac{p_i}{1-p_i} = x_i^\\top \\beta$, luego
                $$ p_i = \\frac{e^{x_i^\\top \\beta}}{1 + e^{x_i^\\top \\beta}}, $$
                y el <strong>odds ratio</strong> $e^{\\beta_j}$ es el cambio multiplicativo en la <em>chance</em>
                $p/(1-p)$ por unidad de $x_j$. Se estima por máxima verosimilitud (sin forma cerrada).</p>
                <p><strong>Poisson</strong> (conteos, liga log): $\\log \\mu_i = x_i^\\top \\beta$; la
                <strong>razón de tasas</strong> es $e^{\\beta_j}$. Requiere equidispersión
                ($\\mathrm{Var} = \\mu$); con <em>sobredispersión</em> use cuasi-Poisson o binomial negativa.</p>
                <p>También: robusta (Huber-White), cuantílica (mínimos absolutos), paso a paso, regularizada
                (ridge $\\ell_2$ / lasso $\\ell_1$), no lineal, variables instrumentales y modelos mixtos.
                Interprete siempre en la escala de la liga (odds ratios o razones de tasas), no como coeficientes
                lineales.</p>
                """,
            ),
            (
                "ANOVA",
                "anova",
                """
                <p>El análisis de la varianza descompone la variabilidad total en la <em>entre grupos</em> y la
                <em>dentro de grupos</em>:
                $$ \\mathrm{SST} = \\mathrm{SSB} + \\mathrm{SSW}, \\qquad
                \\sum_{j}\\sum_{i}(y_{ij}-\\bar{y})^2 = \\sum_j n_j(\\bar{y}_j-\\bar{y})^2 + \\sum_j\\sum_i (y_{ij}-\\bar{y}_j)^2. $$
                Con $k$ grupos y $n$ totales, el estadístico es
                $$ F = \\frac{\\mathrm{SSB}/(k-1)}{\\mathrm{SSW}/(n-k)} \\sim F_{k-1,\\,n-k} \\ \\text{bajo } H_0. $$
                Un $F$ grande indica que las medias difieren más de lo esperable por azar.</p>
                <p><strong>Supuestos.</strong> Independencia, normalidad y homogeneidad de varianzas de los grupos.
                Si se rechaza $H_0$, un <strong>post-hoc</strong> (Tukey, Bonferroni, Games-Howell) identifica qué
                pares difieren; el tamaño del efecto es $\\eta^2 = \\mathrm{SSB}/\\mathrm{SST}$ o el $\\omega^2$
                (insesgado). Un ANOVA significativo solo dice «al menos un grupo difiere».</p>
                """,
            ),
            (
                "ANOVA avanzado",
                "anova_avanzado",
                """
                <p><strong>ANCOVA</strong> añade una covariable continua al ANOVA y compara medias <em>ajustadas</em>
                por ella. <strong>Medidas repetidas</strong> modela la correlación entre mediciones del mismo sujeto
                y exige <em>esfericidad</em> (prueba de Mauchly, correcciones Greenhouse-Geisser o Huynh-Feldt).
                <strong>MANOVA</strong> analiza varias respuestas simultáneas con trazas de Pillai, Wilks, Hotelling
                o Roy. Los <strong>modelos mixtos</strong> combinan efectos fijos y aleatorios
                ($y = X\\beta + Zb + \\varepsilon$).</p>
                <p>Elegir la estructura correcta (intra vs. entre sujetos, efectos aleatorios) es esencial: un error
                de especificación infla el error tipo I o pierde potencia.</p>
                """,
            ),
            (
                "Supervivencia y confiabilidad",
                "supervivencia",
                """
                <p>Sea $T$ el tiempo hasta el evento con función de supervivencia $S(t) = P(T > t)$, densidad $f$ y
                riesgo $h(t) = f(t)/S(t)$ (la tasa instantánea de fallo entre los que aún están en riesgo).</p>
                <p><strong>Kaplan-Meier.</strong> Con tiempos de evento ordenados $t_{(j)}$ y $d_j$ eventos en riesgo
                $n_j$,
                $$ \\hat{S}(t) = \\prod_{j: t_{(j)} \\le t} \\left(1 - \\frac{d_j}{n_j}\\right). $$
                La censura (casos sin evento al cerrar el estudio) se maneja de forma natural: entran en $n_j$ hasta
                su censura. La comparación de grupos usa el <strong>log-rank</strong>.</p>
                <p><strong>Cox.</strong> Modelo semiparamétrico de riesgos proporcionales
                $$ h(t \\mid x) = h_0(t)\\, e^{x^\\top \\beta}, $$
                donde $h_0$ es no especificada; el <strong>hazard ratio</strong> $e^{\\beta_j}$ es el riesgo relativo
                instantáneo por unidad de $x_j$. Requiere <em>riesgos proporcionales</em> (curvas log-log paralelas).</p>
                <p><strong>Paramétricos.</strong> Weibull ($h(t) = \\lambda k t^{k-1}$), exponencial ($h$ constante),
                log-normal y log-logística. La mediana de supervivencia ($\\hat{S}=0.5$) es más informativa que la
                media cuando hay censura.</p>
                """ + _fig("kaplan_meier", "Curva de Kaplan-Meier (escalonada); las marcas │ señalan censuras."),
            ),
            (
                "Ecuaciones estructurales (SEM)",
                "sem",
                """
                <p>Modela relaciones causales entre variables observadas y <strong>latentes</strong>. El modelo de
                medida (AFC) es $x = \\Lambda \\xi + \\delta$, con $\\Lambda$ las cargas y $\\xi$ los factores; el
                modelo estructural es $\\eta = B\\eta + \\Gamma\\xi + \\zeta$.</p>
                <p>En el <strong>análisis de rutas</strong> (solo observadas), cada ecuación es una regresión; el
                coeficiente de ruta estandarizado es $\\beta = b\\, s_x/s_y$ (efecto en desviaciones estándar).</p>
                <p>El ajuste se evalúa con $\\chi^2$ y su grados de libertad, y con índices robustos:
                $\\mathrm{CFI}$ y $\\mathrm{TLI}$ (comparan con un modelo independiente; $\\ge 0.95$ es bueno) y
                $\\mathrm{RMSEA} = \\sqrt{\\max(0, (\\chi^2 - \\mathrm{gl})/(\\mathrm{gl}\\cdot n))}$ ($\\le 0.06$
                es bueno). Un buen ajuste no demuestra causalidad: solo que el modelo es <em>compatible</em> con los
                datos.</p>
                """,
            ),
        ],
    ),
    (
        "Diseño y optimización",
        [
            (
                "Diseño de experimentos (DOE)",
                "doe",
                """
                <p>En un <strong>factorial $2^k$</strong>, cada factor toma dos niveles codificados $\\pm 1$; el modelo
                $$ y = \\beta_0 + \\sum_j \\beta_j x_j + \\sum_{j<l} \\beta_{jl} x_j x_l + \\cdots + \\varepsilon $$
                estima efectos principales e interacciones. El <strong>efecto principal</strong> de un factor es la
                diferencia de la respuesta media entre sus dos niveles, promediada sobre los demás:
                $E_j = \\bar{y}_{x_j=+1} - \\bar{y}_{x_j=-1}$.</p>
                <p>La <strong>superficie de respuesta</strong> (diseño compuesto central) añade puntos axiales y
                centrales para ajustar un modelo cuadrático y localizar el óptimo. La <strong>aleatorización</strong>
                controla variables no consideradas y las <strong>réplicas</strong> aportan el error experimental.
                Un efecto es relevante si es estadísticamente significativo y de magnitud prácticamente importante.</p>
                """,
            ),
            (
                "Optimización (programación lineal y entera)",
                "optimizacion",
                """
                <p>Problema lineal: maximizar (o minimizar) $c^\\top x$ sujeto a $Ax \\le b$, $x \\ge 0$. El óptimo,
                si existe, está en un vértice del poliedro factible (método <em>simplex</em>). La versión
                <strong>entera</strong> (MILP) añade $x_j \\in \\mathbb{Z}$ o $x_j \\in \\{0,1\\}$ y se resuelve por
                <em>branch-and-bound</em> con relajación lineal.</p>
                <p>La <strong>dualidad</strong> da una cota del óptimo y los <em>precios sombra</em> (cuánto mejora el
                objetivo por unidad de recurso). El resultado incluye el valor objetivo óptimo, la solución y la
                <strong>holgura</strong> de cada restricción; verifique siempre la factibilidad.</p>
                """,
            ),
        ],
    ),
    (
        "Series y econometría",
        [
            (
                "Series de tiempo",
                "series",
                """
                <p>Una serie $\\{y_t\\}$ se descompone en tendencia, estacionalidad, ciclo y residuo. Es
                <em>estacionaria</em> si media y autocovarianza $\\gamma_h = \\mathrm{Cov}(y_t, y_{t-h})$ no dependen
                de $t$; la función de autocorrelación $\\rho_h = \\gamma_h/\\gamma_0$ y la autocorrelación parcial
                identifican la estructura.</p>
                <p><strong>ARIMA</strong> combina autorregresivo $\\phi(B)$, integración $\\nabla^d$ y media móvil
                $\\theta(B)$: $\\phi(B)(1-B)^d y_t = \\theta(B)\\varepsilon_t$. La estacionariedad exige raíces de
                $\\phi$ fuera del círculo unidad (pruebas ADF, KPSS). El <strong>suavizamiento exponencial</strong>
                (Holt-Winters, ETS) pondera decrecientemente el pasado. El pronóstico se evalúa con MAE, RMSE, MAPE
                y MASE sobre un conjunto de validación; una serie no estacionaria produce regresiones espurias.</p>
                """,
            ),
            (
                "Econometría",
                "econometria",
                """
                <p><strong>VAR</strong>: $y_t = A_1 y_{t-1} + \\cdots + A_p y_{t-p} + \\varepsilon_t$; la
                <strong>causalidad de Granger</strong> prueba si los rezagos de una variable ayudan a predecir otra;
                el <strong>impulso-respuesta</strong> y la descomposición de varianza describen la dinámica.</p>
                <p><strong>Cointegración</strong>: dos series $I(1)$ están cointegradas si existe $\\beta$ tal que
                $\\beta^\\top y_t$ es $I(0)$ (equilibrio de largo plazo); el <strong>VECM</strong> separa el ajuste de
                corto plazo del equilibrio.</p>
                <p><strong>GARCH</strong>: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$
                (persistencia $\\alpha + \\beta$) modela la volatilidad condicional. Los <strong>datos de panel</strong>
                ($y_{it} = x_{it}^\\top\\beta + \\alpha_i + \\varepsilon_{it}$) eligen efectos fijos o aleatorios
                (Hausman); las <strong>variables instrumentales</strong> resuelven la endogeneidad ($E[x\\varepsilon]
                \\neq 0$) con instrumentos $Z$ correlacionados con $x$ y exógenos.</p>
                """,
            ),
        ],
    ),
    (
        "Multivariado y aprendizaje automático",
        [
            (
                "Multivariado",
                "multivariado",
                """
                <p><strong>PCA</strong> busca direcciones de máxima varianza: los componentes son los vectores
                propios de la matriz de covarianza $S$, ordenados por sus valores propios $\\lambda_1 \\ge \\cdots
                \\ge \\lambda_p$; el componente $k$ explica la fracción $\\lambda_k/\\sum_j \\lambda_j$ de la varianza.
                Estandarice las variables (escalas distintas) y retenga componentes con criterio objetivo (autovalor
                $>1$, codo, varianza acumulada).</p>
                <p><strong>Análisis factorial</strong> postula $x = \\Lambda f + u$ con factores $f$ y unicidades $u$.
                El <strong>clúster</strong> (k-medias minimiza la varianza intra-grupo; jerárquico aglomera por
                distancia) se valida con la <strong>silueta</strong>. El <strong>discriminante</strong> clasifica por
                verosimilitud; las <strong>correspondencias</strong> resumen tablas de contingencia; la
                <strong>fiabilidad</strong> usa el $\\alpha$ de Cronbach.</p>
                """,
            ),
            (
                "Aprendizaje automático",
                "aprendizaje",
                """
                <p>Modelos predictivos evaluados por <strong>validación cruzada</strong> (k-pliegues): el error se
                estima en datos no vistos, evitando el <strong>sobreajuste</strong> (ajuste perfecto al entrenamiento
                pero pobre generalización). Métricas de clasificación: exactitud, precisión, sensibilidad, F1 y el
                <strong>área bajo la curva ROC</strong> (AUC), robusta al desbalance.</p>
                <p>Los <strong>árboles</strong> particionan el espacio con reglas; los <strong>bosques</strong> y el
                <strong>boosting</strong> combinan muchos árboles (bagging vs. ajuste secuencial de residuos); el
                <strong>SVM</strong> maximiza el margen de separación; las <strong>redes</strong> ajustan pesos por
                descenso de gradiente. Compare siempre contra una línea base y reporte la media y dispersión de la
                métrica entre pliegues.</p>
                """ + _fig("roc", "Curva ROC: sensibilidad frente a 1 − especificidad; el área (AUC) mide la capacidad de discriminación."),
            ),
        ],
    ),
    (
        "Calidad y gráficos",
        [
            (
                "Control de calidad (SPC)",
                "calidad",
                """
                <p>Un proceso está <strong>bajo control estadístico</strong> si su variación es solo <em>común</em>
                (inherente). Las cartas de control trazan un estadístico con límites $\\mu \\pm 3\\sigma$ (3 desviaciones
                estándar del estadístico); un punto fuera de límites o un patrón (rachas, tendencias) indica una
                <em>causa asignable</em>.</p>
                <p>La <strong>capacidad</strong> compara la variación del proceso con la especificación
                ($\\mathrm{USL} - \\mathrm{LSL}$):
                $$ C_p = \\frac{\\mathrm{USL} - \\mathrm{LSL}}{6\\sigma}, \\qquad
                C_{pk} = \\min\\left(\\frac{\\mathrm{USL} - \\mu}{3\\sigma}, \\frac{\\mu - \\mathrm{LSL}}{3\\sigma}\\right). $$
                $C_p$ mide la dispersión y $C_{pk}$ también el centrado; valores $\\ge 1.33$ suelen considerarse
                aceptables. El <strong>Gage R&amp;R</strong> separa la variación del sistema de medición
                (repetibilidad = equipo, reproducibilidad = operadores) de la de las piezas; %GRR $< 10\\%$ es
                aceptable. Primero lleve el proceso a control y solo después evalúe la capacidad.</p>
                """ + _fig("control", "Carta de control: línea central y límites ±3σ; un punto fuera indica causa asignable."),
            ),
            (
                "Gráficos",
                "graficos",
                """
                <p>Cada gráfico responde una pregunta:</p>
                <ul>
                  <li><strong>Histograma</strong>: densidad estimada (anchura de clase elegida por regla de
                  Sturges/Freedman-Diaconis).</li>
                  <li><strong>Caja</strong>: cuartiles, IQR y atípicos ($> Q_3 + 1.5\\,\\mathrm{IQR}$ o $< Q_1 - 1.5\\,\\mathrm{IQR}$).</li>
                  <li><strong>Dispersión</strong>: relación entre dos numéricas (tendencia y forma).</li>
                  <li><strong>Q-Q</strong>: cuantiles muestrales frente a teóricos (normalidad si es recta).</li>
                  <li><strong>Matriz de dispersión / mapa de calor</strong>: correlaciones entre muchas variables.</li>
                </ul>
                """ + _fig("boxplot", "Anatomía de un diagrama de caja: cuartiles, IQR y atípico."),
            ),
            (
                "Transformar datos",
                "transformar",
                """
                <p>Preparación de datos: calcular/recodificar, filtrar, imputar valores faltantes (media, mediana,
                regresión), estandarizar ($z = (x-\\bar{x})/s$), transformar ($\\log$, raíz, diferencias), unir y
                pivotar. La transformación debe ser <em>justificable</em> (linealizar, estabilizar varianza, simetrizar)
                y <em>reproducible</em>; compruebe que las columnas resultantes tengan los tipos correctos.</p>
                """,
            ),
        ],
    ),
]

ABOUT = """
<p><strong>Sigma</strong> es una plataforma web de análisis estadístico: importe datos, ejecute pruebas
rigurosas, genere gráficos e informes y conserve los resultados, todo ejecutado y guardado en su propio
servidor, de forma privada por cuenta.</p>
<p>Esta ayuda reúne una <strong>guía de uso</strong> y una <strong>referencia teórica</strong> escrita con
notación matemática (fórmulas renderizadas con KaTeX) y figuras ilustrativas. Para equivalencias con otros
programas (Minitab, SPSS, EViews, SAS) consulte la guía de equivalencias incluida con el proyecto.</p>
"""
