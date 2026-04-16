from RAG import splitter,loadPDF
import asyncio
import pprint

# async def ma

# asyncio.run(main())


stuff = loadPDF("doc.pdf")
stuff2 = splitter(stuff)
print("\n\n\n\n\n")


pprint.pprint(stuff2)