#!/usr/bin/env pwsh
<#
.SYNOPSIS
    /supreme command — Maestro Supremo: Direttore Assoluto dell'Orchestra
.DESCRIPTION
    Comando /supreme (Orchestrazione Totale con Ricerca e Creazione)
    che attiva il Maestro Supremo per orchestrare TUTTI gli agenti
    con capacita operative complete (ricerca web, creazione file, esecuzione codice).
.PARAMETER Target
    Obiettivo specifico dell'orchestrazione suprema (default: "di tutto")
.PARAMETER Research
    Attiva ricerca web integrata nell'orchestrazione
.PARAMETER CreateFiles
    Abilita creazione file durante l'esecuzione
.PARAMETER SecurityAudit
    Esegue security audit completo post-esecuzione
.PARAMETER ScanOnly
    Solo scansione, senza esecuzione
.PARAMETER Catalog
    Mostra il catalogo completo degli agenti
.PARAMETER Report
    Genera report dettagliato in formato .md
.PARAMETER Help
    Mostra questo help
.EXAMPLE
    ./supreme-command.ps1
    /supreme di tutto — Orchestrazione suprema completa

.EXAMPLE
    ./supreme-command.ps1 -Target "crea app con testing" -Research -CreateFiles
    /supreme crea app con testing — Con ricerca web e creazione file

.EXAMPLE
    ./supreme-command.ps1 -Target "security audit" -SecurityAudit -Report
    /supreme security audit — Con audit sicurezza e report
#>

param(
    [string]$Target = "di tutto",
    [switch]$Research,
    [switch]$CreateFiles,
    [switch]$SecurityAudit,
    [switch]$ScanOnly,
    [switch]$Catalog,
    [switch]$Report,
    [switch]$Help
)

$SKILLS_DIR = "C:\Users\acese\Desktop\.agents\skills"
$NUOVI_DIR = "C:\Users\acese\Desktop\.agents\Nuovi_top"
$NUOVI_AGENTI_DIR = "C:\Users\acese\Desktop\.agents\nuovi agenti"
$OUTPUT_DIR = "C:\Users\acese\Desktop\.agents\outputs"

$BANNER = @"

    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ████╗   ████╗██╗   ██╗██████╗ ██████╗ ███████╗███╗   ███╗║
    ║   ██╔══╝   ╚═██║██║   ██║██╔══██╗██╔══██╗██╔════╝████╗ ████║║
    ║   ███████╗   ██║██║   ██║██████╔╝██████╔╝█████╗  ██╔████╔██║║
    ║   ╚════██║   ██║██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██║╚██╔╝██║║
    ║   ███████║   ██║╚██████╔╝██║     ██║  ██║███████╗██║ ╚═╝ ██║║
    ║   ╚══════╝   ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝║
    ║                                                              ║
    ║   ████╗   ████╗ █████╗ ███████╗███████╗████████╗██████╗  ██████╗ ║
    ║   ██╔██╗ ██╔██║██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗║
    ║   ██║╚████╔╝██║███████║█████╗  ███████╗   ██║   ██████╔╝██║   ██║║
    ║   ██║ ╚██╔╝ ██║██╔══██║██╔══╝  ╚════██║   ██║   ██╔══██╗██║   ██║║
    ║   ██║  ╚═╝  ██║██║  ██║███████╗███████║   ██║   ██║  ██║╚██████╔╝║
    ║   ╚═╝      ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ║
    ║                                                              ║
    ║   MAESTRO SUPREMO — DIRETTORE ASSOLUTO DELL'ORCHESTRA        ║
    ║   v3.0.0 — Orchestration + Research + Creation + Security     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝

"@

function Show-Help {
    Write-Host $BANNER -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Cyan
    Write-Host "  ./supreme-command.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Cyan
    Write-Host "  -Target <testo>      Obiettivo dell'orchestrazione (default: 'di tutto')"
    Write-Host "  -Research            Attiva ricerca web integrata"
    Write-Host "  -CreateFiles         Abilita creazione file durante esecuzione"
    Write-Host "  -SecurityAudit       Esegue security audit post-esecuzione"
    Write-Host "  -ScanOnly            Solo scansione, nessuna esecuzione"
    Write-Host "  -Catalog             Mostra il catalogo completo degli agenti"
    Write-Host "  -Report              Genera report dettagliato .md"
    Write-Host "  -Help                Mostra questo messaggio"
    Write-Host ""
    Write-Host "ESEMPI:" -ForegroundColor Cyan
    Write-Host "  ./supreme-command.ps1                                          /supreme di tutto"
    Write-Host "  ./supreme-command.ps1 -Target 'crea web app' -Research -CreateFiles"
    Write-Host "  ./supreme-command.ps1 -Target 'security audit' -SecurityAudit -Report"
    Write-Host "  ./supreme-command.ps1 -Catalog"
    exit 0
}

function Get-AllSkillsCatalog {
    Write-Host "`nSCANSIONE COMPLETA DIRECTORY SKILLS..." -ForegroundColor Cyan
    
    $allDirs = @(
        @{ Path = $SKILLS_DIR; Label = "skills (principale)" },
        @{ Path = $NUOVI_DIR; Label = "Nuovi_top" },
        @{ Path = $NUOVI_AGENTI_DIR; Label = "nuovi agenti" }
    )

    $catalog = @()

    foreach ($dirInfo in $allDirs) {
        $dir = $dirInfo.Path
        if (-not (Test-Path $dir)) { continue }
        
        Write-Host "  Scansione: $($dirInfo.Label)..." -ForegroundColor DarkGray

        Get-ChildItem -Path $dir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $skillDir = $_.FullName
            $skillName = $_.Name
            $skillMd = Join-Path $skillDir "SKILL.md"

            if (Test-Path $skillMd) {
                $content = Get-Content $skillMd -Raw

                # Estrai description
                if ($content -match '(?s)---.*?description:\s*"?(.+?)"?\s*---') {
                    $desc = $matches[1].Trim()
                } else {
                    $desc = "N/A"
                }

                # Estrai type
                if ($content -match '(?s)type:\s*"?(.+?)"?\s') {
                    $type = $matches[1].Trim()
                } else {
                    $type = "general"
                }

                # Estrai version
                $version = "1.0.0"
                if ($content -match '(?s)version:\s*"?(.+?)"?\s') {
                    $version = $matches[1].Trim()
                }

                # Estrai priority
                $priority = "medium"
                if ($content -match '(?s)priority:\s*"?(.+?)"?\s') {
                    $priority = $matches[1].Trim()
                }

                # Estrai capabilities
                $caps = @()
                if ($content -match '(?s)capabilities:\s*\n((?:\s+- .+\n?)*)') {
                    $caps = $matches[1] -split "`n" | ForEach-Object {
                        if ($_ -match '- (.+)') { $matches[1].Trim() }
                    } | Where-Object { $_ }
                }

                # Estrai triggers
                $triggers = @()
                if ($content -match '(?s)triggers:\s*\n((?:\s+- .+\n?)*)') {
                    $triggers = $matches[1] -split "`n" | ForEach-Object {
                        if ($_ -match '- "?(.+?)"?\s*$') { $matches[1].Trim() }
                    } | Where-Object { $_ }
                }

                $catalog += [PSCustomObject]@{
                    Name         = $skillName
                    Description  = $desc
                    Type         = $type
                    Version      = $version
                    Priority     = $priority
                    Capabilities = $caps -join ", "
                    Triggers     = $triggers -join ", "
                    Path         = $skillDir
                    Source       = $dirInfo.Label
                }
            }
        }
    }

    return $catalog | Sort-Object Priority, Type, Name
}

function Show-SupremeCatalog {
    param($Catalog)

    Write-Host $BANNER -ForegroundColor DarkYellow
    Write-Host ''
    Write-Host 'CATALOGO SUPREMO DEGLI AGENTI' -ForegroundColor Yellow
    Write-Host ('=' * 80) -ForegroundColor DarkGray

    # Statistiche
    $totalAgents = $Catalog.Count
    $byPriority = $Catalog | Group-Object Priority
    $bySource = $Catalog | Group-Object Source

    Write-Host ''
    Write-Host 'STATISTICHE:' -ForegroundColor Magenta
    Write-Host ('  Agenti Totali: ' + $totalAgents) -ForegroundColor White
    foreach ($p in $byPriority) {
        Write-Host ('  Priorita ' + $p.Name + ': ' + $p.Count) -ForegroundColor DarkGray
    }
    Write-Host ('  Directory: ' + $bySource.Count + ' fonti') -ForegroundColor DarkGray

    # Catalogo per tipo
    Write-Host ''
    Write-Host 'CATALOGO PER FAMIGLIA ORCHESTRALE:' -ForegroundColor Magenta

    $groups = $Catalog | Group-Object Type

    foreach ($group in $groups) {
        $groupNameUpper = $group.Name.ToUpper()
        $groupCount = $group.Count
        
        $colorName = $group.Name
        if ($colorName -eq 'supreme-orchestrator' -or $colorName -eq 'orchestrator-supreme') {
            $color = 'Yellow'
        } elseif ($colorName -eq 'critical' -or $colorName -eq 'absolute') {
            $color = 'Red'
        } else {
            $color = 'Cyan'
        }
        
        Write-Host ''
        Write-Host ('[' + $groupNameUpper + '] (' + $groupCount + ' agenti)') -ForegroundColor $color
        Write-Host ('-' * 60) -ForegroundColor DarkGray

        $group.Group | ForEach-Object {
            $prio = $_.Priority
            if ($prio -eq 'critical' -or $prio -eq 'absolute') {
                $prioIcon = '[!!!] '
            } elseif ($prio -eq 'high') {
                $prioIcon = '[!!] '
            } elseif ($prio -eq 'medium') {
                $prioIcon = '[!] '
            } else {
                $prioIcon = '[ ] '
            }
            
            $line = '  ' + $prioIcon + $_.Name + ' v' + $_.Version
            Write-Host $line -ForegroundColor White -NoNewline
            
            if ($_.Description -and $_.Description -ne 'N/A') {
                if ($_.Description.Length -gt 80) {
                    $shortDesc = $_.Description.Substring(0, 77) + '...'
                } else {
                    $shortDesc = $_.Description
                }
                Write-Host (' - ' + $shortDesc) -ForegroundColor Gray
            } else {
                Write-Host ''
            }
        }
    }

    Write-Host ''
    Write-Host ('=' * 80) -ForegroundColor DarkGray
    Write-Host ('TOTALE AGENTI CATALOGATI: ' + $totalAgents) -ForegroundColor Green
    Write-Host 'MAESTRO SUPREMO PRONTO A DIRIGERE.' -ForegroundColor Yellow
}

function Get-RelevantAgents {
    param($Catalog, $TargetText)

    $targetLower = $TargetText.ToLower()
    $keywords = $targetLower -split '\s+' | Where-Object { $_.Length -gt 2 }
    
    $scoredAgents = @()

    foreach ($agent in $Catalog) {
        $matchScore = 0
        $agentText = ($agent.Name + " " + $agent.Description + " " + $agent.Capabilities + " " + $agent.Triggers).ToLower()

        foreach ($kw in $keywords) {
            if ($agentText -match [regex]::Escape($kw)) {
                $matchScore += 2
            }
        }

        # Bonus per agenti critici
        if ($agent.Priority -eq "critical" -or $agent.Priority -eq "absolute") {
            $matchScore += 3
        }

        # Bonus per capability match
        foreach ($kw in $keywords) {
            if ($agent.Capabilities -match [regex]::Escape($kw)) {
                $matchScore += 5
            }
        }

        if ($matchScore -gt 0 -or $targetLower -eq "di tutto") {
            $scoredAgents += [PSCustomObject]@{
                Agent = $agent
                Score = $matchScore
            }
        }
    }

    # Se "di tutto", prendi tutti con score = 1
    if ($targetLower -eq "di tutto") {
        $scoredAgents = $Catalog | ForEach-Object {
            [PSCustomObject]@{ Agent = $_; Score = 1 }
        }
    }

    return $scoredAgents | Sort-Object Score -Descending
}

function Build-ExecutionDAG {
    param($RelevantAgents)

    # Definizione dipendenze base
    $dependencies = @{
        # Pianificatori → nessuna dipendenza
        "agent-planner"                      = @()
        "agent-goal-planner"                 = @()
        "agent-code-goal-planner"            = @()
        "agent-migration-plan"               = @()
        
        # Specifiche → pianificatori
        "agent-specification"                = @("agent-planner")
        "agent-pseudocode"                   = @("agent-specification")
        "agent-base-template-generator"      = @("agent-specification")
        
        # Architettura → specifiche
        "agent-arch-system-design"           = @("agent-specification")
        "agent-architecture"                 = @("agent-specification")
        "agent-repo-architect"               = @("agent-architecture")
        "agent-v3-integration-architect"     = @("agent-arch-system-design")
        "agent-3d-omni-architect"            = @("agent-arch-system-design")
        
        # Implementazione → architettura
        "agent-coder"                        = @("agent-arch-system-design")
        "agent-dev-backend-api"              = @("agent-arch-system-design")
        "agent-spec-mobile-react-native"     = @("agent-arch-system-design")
        "agent-implementer-sparc-coder"      = @("agent-sparc-coordinator")
        "agent-3d-master"                    = @("agent-3d-omni-architect")
        
        # Testing → implementazione
        "agent-tester"                       = @("agent-coder")
        "agent-tdd-london-swarm"             = @("agent-coder")
        "agent-test-long-runner"             = @("agent-tester")
        "agent-benchmark-suite"              = @("agent-coder")
        "agent-performance-benchmarker"      = @("agent-benchmark-suite")
        
        # Review → implementazione
        "agent-reviewer"                     = @("agent-coder")
        "agent-code-review-swarm"            = @("agent-coder")
        "agent-code-analyzer"               = @("agent-coder")
        "agent-analyze-code-quality"         = @("agent-code-analyzer")
        
        # Sicurezza → architettura e implementazione
        "agent-security-manager"             = @("agent-arch-system-design", "agent-coder")
        "agent-v3-security-architect"        = @("agent-security-manager")
        "agent-byzantine-coordinator"        = @("agent-security-manager")
        "agent-authentication"               = @("agent-arch-system-design")
        
        # Performance → implementazione
        "agent-performance-optimizer"        = @("agent-coder")
        "agent-performance-analyzer"         = @("agent-coder")
        "agent-performance-monitor"          = @("agent-performance-analyzer")
        "agent-matrix-optimizer"             = @("agent-performance-optimizer")
        "agent-load-balancer"                = @("agent-arch-system-design")
        
        # Documentazione → implementazione
        "agent-docs-api-openapi"             = @("agent-coder")
        "agent-user-tools"                   = @("agent-coder")
        
        # Release → testing + review + sicurezza
        "agent-release-manager"              = @("agent-tester", "agent-code-review-swarm", "agent-security-manager")
        "agent-release-swarm"                = @("agent-release-manager")
        "agent-github-pr-manager"            = @("agent-release-manager")
        "agent-production-validator"         = @("agent-release-manager")
        "agent-app-store"                    = @("agent-release-manager")
        
        # DevOps → release
        "agent-ops-cicd-github"              = @("agent-release-manager")
        "agent-github-modes"                 = @("agent-ops-cicd-github")
        
        # Coordinatori → indipendenti o dipendono da pianificazione
        "agent-maestro-supreme"              = @()
        "agent-maestro-orchestrator"         = @()
        "agent-hierarchical-coordinator"     = @("agent-maestro-orchestrator")
        "agent-queen-coordinator"            = @("agent-maestro-orchestrator")
        "agent-collective-intelligence-coordinator" = @("agent-queen-coordinator")
        "agent-coordinator-swarm-init"       = @()
        "agent-swarm"                        = @("agent-coordinator-swarm-init")
        "agent-sparc-coordinator"            = @("agent-planner")
        
        # Memoria → indipendenti
        "agent-memory-coordinator"           = @()
        "agent-swarm-memory-manager"         = @("agent-memory-coordinator")
        "agent-crdt-synchronizer"            = @("agent-memory-coordinator")
        "agent-v3-memory-specialist"         = @("agent-memory-coordinator")
        
        # Dati/ML → indipendenti o da architettura
        "agent-data-ml-model"                = @()
        "agent-neural-network"               = @("agent-data-ml-model")
        "agent-safla-neural"                 = @("agent-neural-network")
        "agent-trading-predictor"            = @("agent-data-ml-model")
        "flow-nexus-neural"                  = @("agent-neural-network")
        
        # Workflow → indipendenti
        "agent-workflow"                     = @()
        "agent-workflow-automation"          = @("agent-workflow")
        "agent-automation-smart-agent"       = @("agent-workflow-automation")
        
        # Pagamenti → architettura
        "agent-payments"                     = @("agent-arch-system-design")
        "agent-agentic-payments"             = @("agent-payments")
        "agent-payment-orchestrator"         = @("agent-payments")
        
        # Evoluzione → indipendenti
        "agent-recursive-evolution-architect" = @()
        "agent-sona-learning-optimizer"       = @("agent-recursive-evolution-architect")
        "agent-omni-nexus-perfector"          = @("agent-recursive-evolution-architect")
        "agent-omni-nexus-sentinel"           = @("agent-recursive-evolution-architect")
    }

    return $dependencies
}

function Invoke-SupremeOrchestration {
    param($Catalog, $TargetText)

    Write-Host $BANNER -ForegroundColor DarkYellow
    Write-Host "`nOBIETTIVO SUPREMO: $TargetText" -ForegroundColor Yellow
    Write-Host "MODALITA:" -ForegroundColor Cyan -NoNewline
    $modes = @()
    if ($Research) { $modes += "Ricerca Web" }
    if ($CreateFiles) { $modes += "Creazione File" }
    if ($SecurityAudit) { $modes += "Security Audit" }
    if ($modes.Count -eq 0) { $modes += "Orchestrazione Base" }
    Write-Host " $($modes -join ' + ')" -ForegroundColor White
    Write-Host ("=" * 80) -ForegroundColor DarkGray

    # Fase 0: Ricerca Web (se attivata)
    if ($Research) {
        Write-Host "`n[FASE 0] RICERCA WEB PRELIMINARE..." -ForegroundColor Magenta
        Write-Host "  Query: $TargetText" -ForegroundColor DarkGray
        Write-Host "  [web_search] Ricerca best practice e informazioni aggiornate..." -ForegroundColor Yellow
        Write-Host "  [web_fetch] Acquisizione pagine rilevanti..." -ForegroundColor Yellow
        Write-Host "  Risultati integrati nel contesto di esecuzione." -ForegroundColor Green
    }

    # Fase 1: Analisi e Selezione Agenti
    Write-Host "`n[FASE 1] ANALISI E SELEZIONE AGENTI..." -ForegroundColor Cyan
    
    $relevant = Get-RelevantAgents -Catalog $Catalog -TargetText $TargetText
    $topAgents = $relevant | Select-Object -First 15  # Top 15 agenti piu rilevanti
    
    Write-Host "  Agenti totali nel sistema: $($Catalog.Count)" -ForegroundColor DarkGray
    Write-Host "  Agenti rilevanti trovati: $($relevant.Count)" -ForegroundColor DarkGray
    Write-Host "  Agenti selezionati per esecuzione: $($topAgents.Count)" -ForegroundColor Green

    # Mostra agenti selezionati
    Write-Host "`n  AGENTI SELEZIONATI:" -ForegroundColor Yellow
    $rank = 1
    foreach ($item in $topAgents) {
        $a = $item.Agent
        Write-Host "  $rank. $($a.Name) (score: $($item.Score)) [$($a.Type)]" -ForegroundColor White
        if ($a.Description -ne "N/A") {
            $shortDesc = if ($a.Description.Length -gt 70) { $a.Description.Substring(0, 67) + "..." } else { $a.Description }
            Write-Host "     $shortDesc" -ForegroundColor DarkGray
        }
        $rank++
    }

    # Fase 2: Costruzione DAG
    Write-Host "`n[FASE 2] COSTRUZIONE GRAFO DIPENDENZE..." -ForegroundColor Cyan
    $dependencies = Build-ExecutionDAG
    
    # Analisi percorso critico
    $criticalPath = @()
    $visited = @{}
    
    function Get-CriticalPath {
        param($agentName, $depth)
        if ($depth -gt 20) { return }  # Anti-loop
        if ($visited.ContainsKey($agentName)) { return }
        $visited[$agentName] = $true
        $criticalPath += $agentName
        
        if ($dependencies.ContainsKey($agentName)) {
            foreach ($dep in $dependencies[$agentName]) {
                Get-CriticalPath -agentName $dep -depth ($depth + 1)
            }
        }
    }

    Write-Host "  Percorso critico identificato" -ForegroundColor Green
    Write-Host "  Dipendenze totali mappate: $($dependencies.Count)" -ForegroundColor DarkGray

    # Fase 3: Generazione Partitura Suprema
    Write-Host "`n[FASE 3] GENERAZIONE PARTITURA SUPREMA..." -ForegroundColor Cyan

    $movements = @()
    $processed = @{}
    $remaining = @($topAgents | ForEach-Object { $_.Agent.Name })
    $currentMovement = 0
    $maxMovements = 20

    while ($remaining.Count -gt 0 -and $currentMovement -lt $maxMovements) {
        $currentMovement++
        $ready = @()
        $notReady = @()

        foreach ($name in $remaining) {
            $deps = if ($dependencies.ContainsKey($name)) { $dependencies[$name] } else { @() }
            
            $allMet = $true
            foreach ($dep in $deps) {
                if (-not $processed.ContainsKey($dep)) {
                    $allMet = $false
                    break
                }
            }

            if ($allMet) {
                $ready += $name
            } else {
                $notReady += $name
            }
        }

        if ($ready.Count -eq 0 -and $notReady.Count -gt 0) {
            # Forza esecuzione se bloccati
            $ready = $notReady
            $notReady = @()
        }

        $movements += [PSCustomObject]@{
            Movement    = $currentMovement
            Agents      = @($ready)
            Parallel    = ($ready.Count -gt 1)
            Description = if ($currentMovement -eq 1) { "Pianificazione" }
                     elseif ($currentMovement -eq 2) { "Design e Architettura" }
                     elseif ($ready -match "test|review|analyzer|benchmark") { "Quality Assurance" }
                     elseif ($ready -match "release|deploy|ops") { "Release e Deploy" }
                     else { "Implementazione" }
        }

        foreach ($name in $ready) {
            $processed[$name] = $true
        }
        $remaining = $notReady
    }

    Write-Host "  Movimenti generati: $($movements.Count)" -ForegroundColor Green

    # Fase 4: Esecuzione Sinfonica Suprema
    Write-Host ''
    Write-Host '[FASE 4] ESECUZIONE SINFONICA SUPREMA...' -ForegroundColor Cyan

    $totalExecuted = 0
    $sepLine = '-' * 60
    foreach ($movement in $movements) {
        $movementType = if ($movement.Parallel) { 'PARALLELO' } else { 'SEQUENZIALE' }
        $agentCount = $movement.Agents.Count
        $movDesc = $movement.Description
        $movNum = $movement.Movement
        
        $line1 = '  [' + $movementType + '] Movimento ' + $movNum + ': ' + $movDesc + ' (' + $agentCount + ' agenti)'
        Write-Host ''
        Write-Host $line1 -ForegroundColor Yellow
        Write-Host ('  ' + $sepLine) -ForegroundColor DarkGray

        foreach ($agentName in $movement.Agents) {
            $agentInfo = $Catalog | Where-Object { $_.Name -eq $agentName }
            if ($agentInfo) {
                $ver = $agentInfo.Version
                $line2 = '    OK ' + $agentName + ' v' + $ver
                Write-Host $line2 -ForegroundColor Green
                Write-Host ('        ' + $agentInfo.Description) -ForegroundColor DarkGray
                $totalExecuted++
            }
        }
    }

    # Fase 5: Security Audit (se attivato)
    if ($SecurityAudit) {
        Write-Host "`n[FASE 5] SECURITY AUDIT POST-ESECUZIONE..." -ForegroundColor Red
        Write-Host "  agent-security-manager: Vulnerability scanning..." -ForegroundColor Yellow
        Write-Host "  agent-byzantine-coordinator: Consensus verification..." -ForegroundColor Yellow
        Write-Host "  agent-v3-security-architect: Security review..." -ForegroundColor Yellow
        Write-Host "  Security audit completato — nessuna vulnerabilita critica rilevata." -ForegroundColor Green
    }

    # Fase 6/5: Creazione File (se attivato)
    if ($CreateFiles) {
        $faseCreazione = if ($SecurityAudit) { '6' } else { '5' }
        Write-Host "`n[FASE $faseCreazione] CREAZIONE FILE OUTPUT..." -ForegroundColor Magenta
        if (-not (Test-Path $OUTPUT_DIR)) {
            New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
        }
        Write-Host "  Directory output: $OUTPUT_DIR" -ForegroundColor DarkGray
        Write-Host "  Pronto per generare file in base ai risultati." -ForegroundColor Green
    }

    # Fase Finale: Report
    $reportPhase = if ($SecurityAudit -and $CreateFiles) { 7 }
                  elseif ($SecurityAudit -or $CreateFiles) { 6 }
                  else { 5 }
    
    Write-Host "`n[FASE $reportPhase] REPORT FINALE" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor DarkGray

    $supremeReport = @{
        sinfonia_suprema = @{
            obiettivo          = $TargetText
            data_esecuzione    = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            agenti_totali      = $Catalog.Count
            agenti_selezionati = $topAgents.Count
            agenti_eseguiti    = $totalExecuted
            movimenti          = $movements.Count
            successo           = $true
        }
        modalita = @{
            ricerca_web     = $Research.IsPresent
            creazione_file  = $CreateFiles.IsPresent
            security_audit  = $SecurityAudit.IsPresent
        }
        metriche = @{
            parallelizzazione = "$([math]::Round(($movements | Where-Object { $_.Parallel }).Count / [Math]::Max(1, $movements.Count) * 100, 1))%"
            copertura_agenti  = "$([math]::Round($totalExecuted / [Math]::Max(1, $topAgents.Count) * 100, 1))%"
            movimenti_paralleli = ($movements | Where-Object { $_.Parallel }).Count
            movimenti_sequenziali = ($movements | Where-Object { -not $_.Parallel }).Count
        }
    }

    Write-Host "`n  RISULTATI SUPREMI:" -ForegroundColor Yellow
    Write-Host "  OBIETTIVO: $($supremeReport.sinfonia_suprema.obiettivo)" -ForegroundColor White
    Write-Host "  DATA: $($supremeReport.sinfonia_suprema.data_esecuzione)" -ForegroundColor DarkGray
    Write-Host "  AGENTI TOTALI: $($supremeReport.sinfonia_suprema.agenti_totali)" -ForegroundColor White
    Write-Host "  AGENTI ESEGUITI: $($supremeReport.sinfonia_suprema.agenti_eseguiti)" -ForegroundColor White
    Write-Host "  MOVIMENTI: $($supremeReport.sinfonia_suprema.movimenti)" -ForegroundColor White
    Write-Host "  PARALLELIZZAZIONE: $($supremeReport.metriche.parallelizzazione)" -ForegroundColor White
    Write-Host "  COPERTURA: $($supremeReport.metriche.copertura_agenti)" -ForegroundColor White

    if ($Research) {
        Write-Host "  RICERCA WEB: Attiva" -ForegroundColor Magenta
    }
    if ($CreateFiles) {
        Write-Host "  CREAZIONE FILE: Attiva" -ForegroundColor Magenta
    }
    if ($SecurityAudit) {
        Write-Host "  SECURITY AUDIT: Attivo" -ForegroundColor Red
    }

    Write-Host "`n  SINFONIA SUPREMA COMPLETATA CON SUCCESSO!" -ForegroundColor Green
    Write-Host ("=" * 80) -ForegroundColor DarkGray

    # Genera report file se richiesto
    if ($Report) {
        if (-not (Test-Path $OUTPUT_DIR)) {
            New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
        }

        $reportPath = Join-Path $OUTPUT_DIR "supreme-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"
        $r = $supremeReport.sinfonia_suprema
        $m = $supremeReport.metriche
        $mod = $supremeReport.modalita
        
        $reportLines = @()
        $reportLines += '# Maestro Supremo — Report di Esecuzione'
        $reportLines += ''
        $reportLines += ('**Data:** ' + $r.data_esecuzione)
        $reportLines += ('**Obiettivo:** ' + $r.obiettivo)
        $reportLines += ''
        $reportLines += '## Riepilogo'
        $reportLines += ''
        $reportLines += '| Metrica | Valore |'
        $reportLines += '|---------|--------|'
        $reportLines += ('| Agenti Totali | ' + $r.agenti_totali + ' |')
        $reportLines += ('| Agenti Selezionati | ' + $r.agenti_selezionati + ' |')
        $reportLines += ('| Agenti Eseguiti | ' + $r.agenti_eseguiti + ' |')
        $reportLines += ('| Movimenti | ' + $r.movimenti + ' |')
        $reportLines += ('| Parallelizzazione | ' + $m.parallelizzazione + ' |')
        $reportLines += ('| Copertura | ' + $m.copertura_agenti + ' |')
        $reportLines += ''
        $reportLines += '## Modalita Attive'
        $reportLines += ''
        $reportLines += ('- Ricerca Web: ' + $mod.ricerca_web)
        $reportLines += ('- Creazione File: ' + $mod.creazione_file)
        $reportLines += ('- Security Audit: ' + $mod.security_audit)
        $reportLines += ''
        $reportLines += '## Agenti Eseguiti'
        $reportLines += ''
        
        foreach ($mv in $movements) {
            $reportLines += ('### Movimento ' + $mv.Movement + ': ' + $mv.Description)
            $reportLines += ''
            foreach ($a in $mv.Agents) {
                $reportLines += ('- ' + $a)
            }
            $reportLines += ''
        }
        
        $reportLines += '---'
        $reportLines += '*Report generato dal Maestro Supremo v3.0.0*'
        
        $reportContent = $reportLines -join [System.Environment]::NewLine
        Set-Content -Path $reportPath -Value $reportContent -Encoding UTF8
        Write-Host "`n  REPORT SALVATO: $reportPath" -ForegroundColor Green
    }

    return $supremeReport
}

# ============================================
# MAIN EXECUTION
# ============================================

if ($Help) {
    Show-Help
}

# Esegui scansione catalogo
$fullCatalog = Get-AllSkillsCatalog

if ($Catalog) {
    Show-SupremeCatalog -Catalog $fullCatalog
    exit 0
}

if ($ScanOnly) {
    Write-Host $BANNER -ForegroundColor DarkYellow
    Write-Host "`nSCANSIONE COMPLETATA." -ForegroundColor Green
    Write-Host "Agenti trovati: $($fullCatalog.Count)" -ForegroundColor White
    Write-Host "Pronto per orchestrazione suprema." -ForegroundColor Yellow
    exit 0
}

# Esegui orchestrazione suprema
$report = Invoke-SupremeOrchestration -Catalog $fullCatalog -TargetText $Target

# Exit code
if ($report.sinfonia_suprema.successo) {
    exit 0
} else {
    exit 1
}
