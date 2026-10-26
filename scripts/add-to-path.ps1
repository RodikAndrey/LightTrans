# Добавляет каталог lighttrans в PATH пользователя (идемпотентно)
# и уведомляет систему (WM_SETTINGCHANGE), чтобы новые терминалы это увидели.
$dir = 'F:\LightTrans\target\release'
$k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
try {
    $cur = [string]$k.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $kind = $k.GetValueKind('Path')
    if ($cur -notlike "*$dir*") {
        $new = if ($cur.EndsWith(';') -or $cur -eq '') { $cur + $dir } else { $cur + ';' + $dir }
        $k.SetValue('Path', $new, $kind)
        Write-Output "добавлено в PATH: $dir"
    } else {
        Write-Output "уже в PATH: $dir"
    }
} finally { $k.Close() }

Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError=true, CharSet=System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
$res = [UIntPtr]::Zero
[Win32.NativeMethods]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$res) | Out-Null
Write-Output 'система уведомлена (новые терминалы подхватят PATH)'
