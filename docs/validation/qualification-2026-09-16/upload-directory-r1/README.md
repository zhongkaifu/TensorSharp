# External upload storage

`TENSORSHARP_UPLOAD_DIR` lets operators keep uploaded media and extracted video frames outside the application deployment. Unset/blank retains the existing `uploads` directory beside the server binary. Logging already supports `TENSORSHARP_LOG_DIR`.

The original HTTP replay failed its final whole-directory check when generated logs/uploads appeared beside the binaries. This setting supports a fresh replay without excluding filenames from integrity verification. **134 server-options tests pass, zero skipped** ([TRX](upload-directory-r2.trx)). The first compile was stopped by xUnit overload-name errors in concurrent GLM test additions; those helper names were corrected before this successful run. No passing HTTP integrity replay with this setting is yet claimed.
