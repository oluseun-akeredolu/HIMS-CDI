[CmdletBinding()]
param([string]$OutputRoot = (Join-Path $PSScriptRoot 's4_s7_recovery_runs'))
$ErrorActionPreference='Stop'
$owner='oluseun-akeredolu'; $name='HIMS-CDI'; $repo="$owner/$name"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$run=(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_UTC')
$out=Join-Path $OutputRoot $run
New-Item -ItemType Directory -Force -Path $out | Out-Null
$headers=@{'User-Agent'='HIMS-CDI-S4-S7-recovery/0.2.16';'Accept'='application/vnd.github+json'}
function GitBlobSha1([byte[]]$b){
  $prefix=[Text.Encoding]::UTF8.GetBytes("blob $($b.Length)`0")
  $all=New-Object byte[] ($prefix.Length+$b.Length)
  [Array]::Copy($prefix,0,$all,0,$prefix.Length); [Array]::Copy($b,0,$all,$prefix.Length,$b.Length)
  $sha=[Security.Cryptography.SHA1]::Create()
  try { ([BitConverter]::ToString($sha.ComputeHash($all))).Replace('-','').ToLowerInvariant() } finally {$sha.Dispose()}
}
$commit=Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$repo/commits/main" -Headers $headers
$items=Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$repo/contents/supplementary_files?ref=$($commit.sha)" -Headers $headers
$inventory=@()
foreach($fileName in @('File S4 - Data_Extraction_Database.zip','File S7 - Included_Studies_Table.zip')){
  $item=$items | Where-Object {$_.name -eq $fileName} | Select-Object -First 1
  if(-not $item){throw "Could not find $fileName in pinned commit $($commit.sha)."}
  $path=Join-Path $out $fileName
  Invoke-WebRequest -Uri $item.download_url -OutFile $path -Headers $headers -UseBasicParsing
  $bytes=[IO.File]::ReadAllBytes($path); $blob=GitBlobSha1 $bytes
  $sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  $ok=($bytes.Length -eq [int]$item.size) -and ($blob -eq $item.sha)
  $inventory+=([pscustomobject]@{file=$fileName;path=$item.path;pinned_commit=$commit.sha;size_expected=[int]$item.size;size_observed=$bytes.Length;git_blob_sha1_expected=$item.sha;git_blob_sha1_observed=$blob;git_blob_and_size_match=$ok;sha256=$sha256;retrieved_at_utc=(Get-Date).ToUniversalTime().ToString('o')})
  if(-not $ok){throw "Hash/size validation failed for $fileName. Bytes are retained in $out and must not be treated as validated."}
}
$manifest=[pscustomobject]@{repository=$repo;branch='main';pinned_commit=$commit.sha;files=$inventory;scope='Current public GitHub main snapshot only; not proof of historical publication source.';next_step='Run the separate S4/S7 parser and resolve every conflicting/unmatched record against original source passages and the original codebook.'}
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $out 'SOURCE_MANIFEST.json') -Encoding UTF8
$inventory | Export-Csv -LiteralPath (Join-Path $out 'SOURCE_HASHES.csv') -NoTypeInformation -Encoding UTF8
Write-Host "PASS: downloaded both archives from pinned commit $($commit.sha) and verified Git blob ID and byte size."
Write-Host "SHA-256 values are recorded in SOURCE_HASHES.csv. Output: $out"
Write-Host "This validates the current public GitHub snapshot only, not the historical archive or any safeguard claim."
