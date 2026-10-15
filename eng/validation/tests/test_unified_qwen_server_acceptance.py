"""Model-free execution of the real PowerShell wrapper's configuration AST.

Only its parameter, audit, child-environment and inherited-override statements
are evaluated. No server launch, model/native hash, network or GPU is used.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


WRAPPER = Path(__file__).resolve().parents[1] / 'unified-qwen-server-acceptance.ps1'
PWSH = shutil.which('pwsh')

# Parsing/evaluating selected original AST statements tests their actual
# PowerShell semantics while leaving the launch/ownership/checker paths idle.
BOOTSTRAP = r"""
$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($env:TS_TEST_WRAPPER,[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count) { throw ($parseErrors | Out-String) }
$try=@($ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.TryStatementAst] })[0]
if(!$try) { throw 'No top-level wrapper try block' }
$body=@($try.Body.Statements)
function IndexOfAssignment($statements,$target) {
    for($i=0;$i -lt $statements.Count;$i++) {
        if($statements[$i] -is [Management.Automation.Language.AssignmentStatementAst] -and
           $statements[$i].Left.Extent.Text -ceq $target) { return $i }
    }
    throw "No assignment to $target"
}
function EvaluateStatements($statements) {
    foreach($statement in $statements) {
        if($statement.Find({param($node)
            $node -is [Management.Automation.Language.CommandAst] -and
            $node.GetCommandName() -eq 'Start-Process'
        },$true)) { throw 'Test refused to evaluate a process launch' }
    }
    . ([scriptblock]::Create(($statements | ForEach-Object { $_.Extent.Text }) -join "`n"))
}
"""


@unittest.skipUnless(PWSH, 'PowerShell7 is required for CPU wrapper AST execution')
class QwenServerAcceptanceConfigurationTests(unittest.TestCase):
    def powershell(self, code):
        environment = dict(os.environ)
        environment['TS_TEST_WRAPPER'] = str(WRAPPER)
        result = subprocess.run([PWSH, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', BOOTSTRAP + code],
                                capture_output=True, text=True, encoding='utf-8', timeout=30, env=environment)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def configuration(self, prefix=False, fused=False, token=False, timing=False):
        values = {'PrefixDiagnostics': prefix, 'FusedTrace': fused, 'TokenTrace': token, 'DiagnosticTiming': timing}
        assignments = '\n'.join(f'${key}=${str(value).lower()}' for key, value in values.items())
        return self.powershell(assignments + r"""
$env:TENSORSHARP_LOG_LEVEL='Warning'
$before=$env:TENSORSHARP_LOG_LEVEL
$exe='no-server-started'; $launchArgs=@('--unified-memory')
$auditIndex=IndexOfAssignment $ast.EndBlock.Statements '$audit'
# Dot-source in this scope so the real hashtable remains visible below.
. ([scriptblock]::Create($ast.EndBlock.Statements[$auditIndex].Extent.Text))
$processOptions=@{}
$first=IndexOfAssignment $body '$audit.childEnvironment'
$after=IndexOfAssignment $body '$owned'
$selected=@($body[$first..($after-1)])
EvaluateStatements $selected
[ordered]@{
    child=$audit.childEnvironment
    processEnvironment=$processOptions.Environment
    parentBefore=$before
    parentAfter=$env:TENSORSHARP_LOG_LEVEL
    diagnostic=$audit.prefixDiagnosticLogging
    selectedContainsParentEnvironmentAssignment=[bool]@($selected | ForEach-Object {
        $_.FindAll({param($node) $node -is [Management.Automation.Language.AssignmentStatementAst] -and
            $node.Left.Extent.Text -match '^\$env:'},$true)
    }).Count
} | ConvertTo-Json -Depth 10 -Compress
""")

    def test_script_parses_and_prefix_diagnostics_is_an_optional_switch(self):
        value = self.powershell(r"""
$parameter=@($ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -ceq 'PrefixDiagnostics' })
if($parameter.Count -ne 1) { throw 'PrefixDiagnostics must occur exactly once' }
[ordered]@{ type=$parameter[0].StaticType.FullName; default=$parameter[0].DefaultValue } | ConvertTo-Json -Compress
""")
        self.assertEqual('System.Management.Automation.SwitchParameter', value['type'])
        self.assertIsNone(value['default'])

    def test_default_does_not_override_child_or_parent_logging(self):
        value = self.configuration()
        self.assertEqual({}, value['child'])
        self.assertIsNone(value['processEnvironment'])
        self.assertFalse(value['diagnostic']['enabled'])
        self.assertFalse(value['diagnostic']['excludesDefaultLoggingPerformanceComparison'])
        self.assertEqual('Warning', value['parentBefore'])
        self.assertEqual('Warning', value['parentAfter'])
        self.assertFalse(value['selectedContainsParentEnvironmentAssignment'])

    def test_prefix_diagnostics_is_child_only_and_explicitly_not_default_performance(self):
        value = self.configuration(prefix=True)
        expected = {'TENSORSHARP_LOG_LEVEL': 'Debug'}
        self.assertEqual(expected, value['child'])
        self.assertEqual(expected, value['processEnvironment'])
        self.assertEqual('Warning', value['parentAfter'])
        self.assertTrue(value['diagnostic']['enabled'])
        self.assertTrue(value['diagnostic']['excludesDefaultLoggingPerformanceComparison'])
        self.assertIn('Do not compare', value['diagnostic']['scope'])
        self.assertIn('without this diagnostic logging', value['diagnostic']['scope'])
        self.assertFalse(value['selectedContainsParentEnvironmentAssignment'])

    def test_prefix_diagnostics_composes_with_existing_trace_configuration(self):
        value = self.configuration(prefix=True, fused=True, token=True, timing=True)
        expected = {'TENSORSHARP_LOG_LEVEL': 'Debug', 'TS_BATCHED_FUSED_TRACE': '1',
                    'TS_VALIDATION_TOKEN_TRACE': '1', 'TS_HOST_MOE_TIMING': '3',
                    'TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS': '1'}
        self.assertEqual(expected, value['child'])
        self.assertEqual(expected, value['processEnvironment'])
        self.assertEqual('Warning', value['parentAfter'])

    def test_inherited_overrides_still_refuse_before_child_configuration(self):
        for prefix in (False, True):
            for inherited in ('TENSORSHARP_LOG_LEVEL', 'TS_HOST_MOE_TIMING', 'GGML_N_THREADS'):
                with self.subTest(prefix=prefix, inherited=inherited):
                    value = self.powershell(f'$PrefixDiagnostics=${str(prefix).lower()}\n' +
                        f'$testOverride={json.dumps(inherited)}\n' + r"""
# Clear only this disposable pwsh process's inherited configuration for isolation.
Get-ChildItem Env: | Where-Object { $_.Name -match '^(TS_|TENSORSHARP_|GGML_|MAX_CONTEXT|MAX_TOKENS|KV_CACHE_|CUDA_VISIBLE|OMP_|BACKEND$)' } |
    ForEach-Object { Remove-Item -LiteralPath ("Env:"+$_.Name) }
[Environment]::SetEnvironmentVariable($testOverride,'unexpected','Process')
$first=IndexOfAssignment $body '$overrides'
$child=IndexOfAssignment $body '$audit.childEnvironment'
function Save($value,$name) { $script:SavedOverrides=@($value); $script:SavedName=$name }
$refused=$null
try { EvaluateStatements @($body[$first..($first+2)]) } catch { $refused=$_.Exception.Message }
[ordered]@{
    refused=$refused
    savedName=$script:SavedName
    savedNames=@($script:SavedOverrides | ForEach-Object Name)
    overrideGuardPrecedesChildConfiguration=($first+2 -lt $child)
} | ConvertTo-Json -Depth 10 -Compress
""")
                    self.assertIn('Inherited configuration override', value['refused'])
                    self.assertEqual('environment-overrides.json', value['savedName'])
                    self.assertEqual([inherited], value['savedNames'])
                    self.assertTrue(value['overrideGuardPrecedesChildConfiguration'])


if __name__ == '__main__':
    unittest.main()
