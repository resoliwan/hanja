import Foundation
import Vision
import ImageIO
import AppKit

let args = CommandLine.arguments
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hant"]
request.usesLanguageCorrection = true
request.automaticallyDetectsLanguage = false
if args.count < 4 {
    let info: [String: Any] = ["revision":request.revision,"supportedLanguages":try request.supportedRecognitionLanguages(),"os":ProcessInfo.processInfo.operatingSystemVersionString,"language":"zh-Hant","correction":true]
    print(String(data:try JSONSerialization.data(withJSONObject:info,options:[.sortedKeys]),encoding:.utf8)!)
    exit(0)
}
let manifest = try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:args[1]))) as! [[String:Any]]
let outputURL=URL(fileURLWithPath:args[3])
if !FileManager.default.fileExists(atPath:outputURL.path) { FileManager.default.createFile(atPath:outputURL.path,contents:nil) }
let existing = try String(contentsOf:outputURL,encoding:.utf8).split(separator:"\n").count
let handle = try FileHandle(forWritingTo:outputURL);try handle.seekToEnd()
for i in existing..<manifest.count {
 autoreleasepool {
    let item=manifest[i];let id=item["id"] as! String;let label=item["label"] as! String
    let t=Date();var errorMessage:String?;var candidates:[String]=[];var observations:[[String:Any]]=[]
    do {
        let url=URL(fileURLWithPath:args[2]).appendingPathComponent(id+".png")
        let handler: VNImageRequestHandler
        if args.count > 4 {
            let source=CGImageSourceCreateWithURL(url as CFURL,nil)!
            let image=CGImageSourceCreateImageAtIndex(source,0,nil)!
            let size=Int(args[4])!
            let context=CGContext(data:nil,width:size,height:size,bitsPerComponent:8,bytesPerRow:size*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.interpolationQuality = .high
            context.draw(image,in:CGRect(x:0,y:0,width:size,height:size))
            handler=VNImageRequestHandler(cgImage:context.makeImage()!,options:[:])
        } else { handler=VNImageRequestHandler(url:url,options:[:]) }
        try handler.perform([request])
        let results=(request.results ?? []).sorted { $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height }
        observations=results.map { o in ["candidates":o.topCandidates(5).map{$0.string},"bbox":[o.boundingBox.origin.x,o.boundingBox.origin.y,o.boundingBox.width,o.boundingBox.height]] }
        candidates=results.first?.topCandidates(5).map{$0.string.trimmingCharacters(in:.whitespacesAndNewlines)} ?? []
    } catch {errorMessage=String(describing:error)}
    let row:[String:Any] = ["id":id,"label":label,"candidates":candidates,"observations":observations,"error":errorMessage as Any? ?? NSNull(),"ms":Date().timeIntervalSince(t)*1000,"top1":candidates.first==label,"top5":candidates.contains(label)]
    do {var data=try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]);data.append(10);try handle.write(contentsOf:data)} catch {fatalError("Output failed: \(error)")}
 }
 if (i+1)%1000==0 || i+1==manifest.count { print("\(i+1) / \(manifest.count)");fflush(stdout) }
}
try handle.close()
