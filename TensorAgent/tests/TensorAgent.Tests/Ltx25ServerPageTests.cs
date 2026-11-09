using System.Text.Json;
using System.Text.RegularExpressions;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

[Collection(ProcessEnvironmentCollection.Name)]
public sealed class Ltx25ServerPageTests
{
    [WebJavaScriptTheory]
    [InlineData("none")]
    [InlineData("image")]
    [InlineData("pair")]
    [InlineData("multiple")]
    [InlineData("longmultiple")]
    [InlineData("identity")]
    [InlineData("identity_then_text")]
    [InlineData("indexed")]
    [InlineData("end")]
    [InlineData("video")]
    [InlineData("audio")]
    public void ActualServerPageRoutesDistilledImageKeyframesAndVideoContinuations(string media)
    {
        var repo=new DirectoryInfo(AppContext.BaseDirectory);
        while(repo is not null&&!File.Exists(Path.Combine(repo.FullName,"TensorSharp.slnx")))repo=repo.Parent;
        Assert.NotNull(repo);
        string page=File.ReadAllText(Path.Combine(repo!.FullName,"TensorSharp.Server.Host","wwwroot","index.html"));
        int start=page.IndexOf("async function runVideoGenerate(",StringComparison.Ordinal);
        int end=page.IndexOf("async function requestAssistantResponse(",start,StringComparison.Ordinal);
        int sendStart=page.IndexOf("async function sendMessage()",StringComparison.Ordinal);
        int sendEnd=page.IndexOf("// Finish the interrupted assistant entry",sendStart,StringComparison.Ordinal);
        string routing=Regex.Match(page,@"const VIDEO_ARCHITECTURES = [^\r\n]+;").Value;
        // The capabilities are what /api/models reports for LTX 2.5 (WebUiChatService), including the
        // 121-frame floor of its identity sheets (Ltx25ReferenceConditioning.MinimumFrames).
        string source=routing+"\n"+page[start..end]+page[sendStart..sendEnd]+"""
            var currentArchitecture='ltx-2.5', currentVideoCaps={supportsImageConditioning:true,supportsEndImageConditioning:true,supportsKeyframeConditioning:true,supportsReferenceConditioning:false,maxReferenceImages:16,defaultPreviewFrames:17,minimumReferenceFrames:121};
            var performance={now:function(){return 0;}};
            var calls=[],bubble={},stats={};
            var built=[], pendingUploadCount=0, pendingAttachments=[], currentLoadedModel='ltx.gguf', activeImageEditTarget=null;
            // The composer's video controls (the page keeps them per loaded model).
            var videoComposerMode='auto', videoComposerFrames=0, videoComposerResolution='default';
            function videoResolutionChoices(){return [{value:'default',label:'Model default'}];}
            function selectedVideoFrames(){return videoComposerFrames>0?videoComposerFrames:17;}
            var messageInput={value:'The woman waves.',style:{}}, attachmentsDiv={}, queuedChatMessages=[], queuedResponseRequested=false;
            function videoImageConditions(){return null;}
            function cancelCurrentGeneration(){}
            function setSendButtonState(){}
            async function drainChatSubmissions(){queuedChatMessages.forEach(function(entry){built.push(entry.message);});queuedChatMessages=[];}
            function ensureCurrentGeneration(){}
            async function fetch(url,options){calls.push({url:url,payload:JSON.parse(options.body)});throw new Error('captured');}
            (async function(){
              var message={content:'a red fox walks'};
              if(__MEDIA__==='image')message.stillImagePaths=['a.png'];
              if(__MEDIA__==='pair')message.stillImagePaths=['a.png','b.png'];
              if(__MEDIA__==='multiple')message.stillImagePaths=['a.png','b.png','c.png'];
              if(__MEDIA__==='longmultiple'){message.stillImagePaths=['a.png','b.png','c.png'];message.videoFrames=97;}
              if(__MEDIA__==='identity'){currentVideoCaps.supportsReferenceConditioning=true;message.stillImagePaths=['a.png','b.png'];message.videoMode='ref';}
              if(__MEDIA__==='identity_then_text'){
                currentVideoCaps.supportsReferenceConditioning=true;videoComposerMode='ref';
                pendingAttachments=[{file:'a.png',mediaType:'image'}];await sendMessage();
                messageInput.value='A red fox runs.';await sendMessage();message=built[1];
              }
              if(__MEDIA__==='indexed')message.imageConditions=[{imagePath:'a.png',frameIndex:5,strength:.65},{imagePath:'b.png',frameIndex:16,strength:1}];
              if(__MEDIA__==='end')message.imageConditions=[{imagePath:'a.png',frameIndex:-1,strength:1}];
              if(__MEDIA__==='video')message.videoFilePaths=['a.mp4'];
              if(__MEDIA__==='audio')message.audioPaths=['a.wav'];
              await runVideoGenerate(message,bubble,stats,{controller:{signal:{}}});
              console.log('<<RESULT>>'+JSON.stringify({routed:VIDEO_ARCHITECTURES.has('ltx-2.5'),calls:calls,status:stats.textContent,built:built}));
            })();
            """.Replace("__MEDIA__",JsonSerializer.Serialize(media),StringComparison.Ordinal);
        string root=Path.Combine(Path.GetTempPath(),"ltx-server-page-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(root);
        try
        {
            var policy=new ExecutionPolicy(true,false,root,Array.Empty<string>(),root){DefaultTimeout=TimeSpan.FromSeconds(10)};
            var context=new InterpreterContext(root,new Dictionary<string,string>(),policy);
            var result=WebJavaScript.Run(source,root,context);Assert.True(result.ExitCode==0,result.Stdout+result.Stderr);
            int at=result.Stdout.IndexOf("<<RESULT>>",StringComparison.Ordinal);Assert.True(at>=0,result.Stdout+result.Stderr);
            using var json=JsonDocument.Parse(result.Stdout[(at+10)..].Trim());var actual=json.RootElement;
            Assert.True(actual.GetProperty("routed").GetBoolean());
            if(media=="audio"){Assert.Empty(actual.GetProperty("calls").EnumerateArray());Assert.Equal("Unsupported input",actual.GetProperty("status").GetString());return;}
            var call=Assert.Single(actual.GetProperty("calls").EnumerateArray());Assert.Equal("/api/video-generate/stream",call.GetProperty("url").GetString());
            var body=call.GetProperty("payload");
            // Identity sheets keep their trained 768x448; every other LTX request leaves the output size to the server/model.
            if(media=="identity"){Assert.Equal(768,body.GetProperty("width").GetInt32());Assert.Equal(448,body.GetProperty("height").GetInt32());}
            else{Assert.False(body.TryGetProperty("width",out _));Assert.False(body.TryGetProperty("height",out _));}
            Assert.Equal(media=="identity"?121:media=="longmultiple"?97:17,body.GetProperty("frames").GetInt32());Assert.Equal(24,body.GetProperty("fps").GetInt32());Assert.Equal(8,body.GetProperty("steps").GetInt32());
            Assert.Equal(1,body.GetProperty("cfg").GetInt32());Assert.Equal(0,body.GetProperty("flowShift").GetInt32());Assert.Equal("euler_ancestral",body.GetProperty("sampler").GetString());Assert.True(body.GetProperty("generateAudio").GetBoolean());
            if(media is "none" or "identity_then_text")Assert.Equal("t2va",body.GetProperty("videoMode").GetString());
            if(media=="identity_then_text")
            {
                var built=actual.GetProperty("built");Assert.Equal(2,built.GetArrayLength());
                Assert.Equal("ref",built[0].GetProperty("videoMode").GetString());
                Assert.False(built[1].TryGetProperty("videoMode",out _));
                Assert.False(built[1].TryGetProperty("stillImagePaths",out _));
            }
            if(media=="image"){Assert.Equal("a.png",body.GetProperty("imagePath").GetString());Assert.Equal("i2v",body.GetProperty("videoMode").GetString());}
            if(media=="video"){Assert.Equal(new[]{"a.mp4"},body.GetProperty("continueFromVideos").EnumerateArray().Select(v=>v.GetString()));Assert.Equal("i2v",body.GetProperty("videoMode").GetString());}
            if(media=="pair"){Assert.Equal("a.png",body.GetProperty("imagePath").GetString());Assert.Equal("b.png",body.GetProperty("endImage").GetString());Assert.Equal("fl2v",body.GetProperty("videoMode").GetString());}
            if(media=="identity"){Assert.Equal("ref",body.GetProperty("videoMode").GetString());Assert.Equal(new[]{"a.png","b.png"},body.GetProperty("referenceImages").EnumerateArray().Select(v=>v.GetString()));Assert.False(body.TryGetProperty("imageConditions",out _));}
            if(media is "multiple" or "longmultiple" or "indexed" or "end")
            {
                Assert.Equal("keyframes",body.GetProperty("videoMode").GetString());
                var conditions=body.GetProperty("imageConditions").EnumerateArray().ToArray();
                Assert.Equal(media=="multiple"?new[]{0,8,16}:media=="longmultiple"?new[]{0,48,96}:media=="indexed"?new[]{5,16}:new[]{-1},conditions.Select(c=>c.GetProperty("frameIndex").GetInt32()));
                if(media=="indexed")Assert.Equal(.65,conditions[0].GetProperty("strength").GetDouble());
            }
        }
        finally{Directory.Delete(root,true);}
    }
}
