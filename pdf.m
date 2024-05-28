#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

#include <CoreFoundation/CFURL.h>
#include <CoreFoundation/CFBase.h>
#include <CoreGraphics/CGImage.h>
#include <CoreGraphics/CGContext.h>
#include <CoreGraphics/CGPDFPage.h>
#include <CoreGraphics/CGBitmapContext.h>

#include "wrapper.h"

pdf_size get_pdf_page_size(int pagenumber, char* filepath) 
{
    CFStringRef path = CFStringCreateWithCString(NULL, filepath, kCFStringEncodingUTF8);

    CFURLRef url;
    CGPDFPageRef page;
    url = CFURLCreateWithFileSystemPath (NULL, path, // 1
            kCFURLPOSIXPathStyle, 0);

    CGPDFDocumentRef document;

    document = CGPDFDocumentCreateWithURL (url);
    page = CGPDFDocumentGetPage (document, pagenumber+1);
    CFRelease(url);

    float dpi = 300.0 / 72.0;

    CGRect bounds = CGPDFPageGetBoxRect(page, kCGPDFMediaBox);
    int w = (int)CGRectGetWidth(bounds) * dpi;
    int h = (int)CGRectGetHeight(bounds) * dpi;

    CGPDFDocumentRelease(document);
    pdf_size size;
    size.width = w;
    size.height = h;
    return size;
}

char* get_pdf_page(int pagenumber, char* filepath) 
{

    CFStringRef path = CFStringCreateWithCString(NULL, filepath, kCFStringEncodingUTF8);

    CFURLRef url;
    CGPDFPageRef page;
    url = CFURLCreateWithFileSystemPath (NULL, path, // 1
            kCFURLPOSIXPathStyle, 0);


    CGPDFDocumentRef document;

    document = CGPDFDocumentCreateWithURL (url);
    page = CGPDFDocumentGetPage (document, pagenumber+1);
    CFRelease(url);

    float dpi = 300.0 / 72.0;

    CGRect bounds = CGPDFPageGetBoxRect(page, kCGPDFMediaBox);
    int w = (int)CGRectGetWidth(bounds) * dpi;
    int h = (int)CGRectGetHeight(bounds) * dpi;

    CGBitmapInfo info = kCGBitmapByteOrder32Big | kCGImageAlphaNoneSkipLast;;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();

    char* bitmap = (char*)malloc(4*w*h);
    memset(bitmap, 0xFF, 4 * w * h);

    CGContextRef ctx = CGBitmapContextCreate(bitmap, w, h, 8, 4*w, cs, info);

    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextScaleCTM(ctx, dpi, dpi);
    CGContextSaveGState(ctx);

    CGContextDrawPDFPage(ctx, page);
    CGContextRestoreGState(ctx);

    CGImageRef image = CGBitmapContextCreateImage(ctx);

    CGColorSpaceRelease(cs);
    CGImageRelease(image);
    CGContextRelease(ctx);
    CGPDFDocumentRelease(document);
    return (char*)bitmap;

}
