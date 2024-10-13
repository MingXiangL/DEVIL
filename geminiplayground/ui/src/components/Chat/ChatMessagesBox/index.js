import CodeCopyBtn from "@/components/CodeCopyButton";
import {Badge} from "@/components/ui/badge";
import {BotIcon, ClipboardCopy, UserCircle2} from "lucide-react";
import Spinner from "@/components/Spinner/Spinner";
import Markdown from "react-markdown";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeStringify from "rehype-stringify";
import rehypeRaw from "rehype-raw";
import {Prism as SyntaxHighlighter} from "react-syntax-highlighter";
import {oneDark as darkTheme, oneLight as lightTheme} from "react-syntax-highlighter/src/styles/prism";
import Image from "next/image";
import {Button} from "@/components/ui/button";
import {TrashIcon} from "@radix-ui/react-icons";
import {useTheme} from "next-themes";

export const Pre = ({children}) => <pre className="blog-pre">
        <CodeCopyBtn>{children}</CodeCopyBtn>
    {children}
    </pre>

const OutputMessage = ({markdown}) => {
    const {theme} = useTheme();
    return <Markdown
        className="markdown"
        remarkPlugins={[remarkParse, remarkGfm, remarkRehype, rehypeStringify]}
        rehypePlugins={[rehypeRaw]}
        components={{
            pre: Pre,
            code({node, inline, className = "blog-code", children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                    <SyntaxHighlighter
                        style={theme === "dark" ? darkTheme : lightTheme}
                        showLineNumbers
                        language={match[1]}
                        PreTag="div"
                        {...props}
                    >
                        {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                ) : (
                    <code className={className} {...props}>
                        {children}
                    </code>
                )
            }
        }}
    >
        {markdown}
    </Markdown>
}


function UserMessage({message: {content, moment}}) {

    return <div className="flex flex-row m-1 p-1">
        <div className="content-center">
            <UserCircle2 className={"size-10"}/>
        </div>
        <div className="flex flex-col rounded-2xl bg-background fade-in mr-2 w-full p-3 divide-y">
            <span className="p-2">{content}</span>
            <div className="flex items-center grid-cols-2 place-content-between">
                <div/>
                <div className="flex items-center gap-3 mt-2">
                    <p className="text-sm font-semibold">User</p>
                    <p className="text-xs text-muted-foreground">{moment}</p>
                </div>
            </div>
        </div>
    </div>;
}

function ModelMessage({message: {content, moment, error, loading = false}}) {

    return (
        <div className="flex m-1 p-1">
            {loading ? (<div className="rounded-2xl bg-background fade-in w-full p-3 mt-2">
                <div className="flex">
                    <Spinner className="place-content-start"/>
                </div>
            </div>) : (
                <div className="rounded-2xl bg-background fade-in mr-2 w-full p-3 divide-y">
                    <div>
                        <div>
                            {<OutputMessage markdown={content}/>}
                        </div>
                        <div>
                            {error &&
                                <p className="text-red-500 text-sm pb-4">{error?.response?.data?.detail || error.message}</p>}
                        </div>
                    </div>
                    <div className="flex items-center grid-cols-2 place-content-between">
                        <div>
                            <Button variant="ghost" onClick={() => navigator.clipboard.writeText(content)}>
                                <ClipboardCopy size="15"/>
                            </Button>
                        </div>
                        <div className="flex items-center gap-3 mt-2">
                            <p className="text-sm font-semibold">Model</p>
                            <p className="text-xs text-muted-foreground">{moment}</p>
                        </div>
                    </div>
                </div>
            )}
            <div className="content-center">
                <BotIcon className="size-10"/>
            </div>
        </div>
    );
}

export default function ChatMessagesBox({messages = [], onClearChatQueue = null}) {

    const emptyMessageQueueContent = <div className="flex flex-col items-center justify-center h-full">
        <Image src="gemini-logo.svg" alt="Gemini Logo" width={200} height={200} priority={true}/>
    </div>
    if (messages.length === 0) {
        return emptyMessageQueueContent;
    }

    const messageQueueContent = [...messages].reverse().map((message, index) => {
        if (message.role === "user") {
            return <UserMessage key={index} message={message}/>
        } else {
            return <ModelMessage key={index} message={message}/>
        }
    });

    function clearHistory() {
        console.log("Clearing chat history");
        if (onClearChatQueue) {
            onClearChatQueue();
        }
    }

    return (
        <div className="relative pt-[40px]">
            <Badge variant="outline" className="absolute left-0 top-3">
                Output
            </Badge>
            <Button className="absolute right-0 top-0" variant="ghost" onClick={clearHistory}>
                <TrashIcon/>clear
            </Button>
            {messageQueueContent.length > 0 ? messageQueueContent : emptyMessageQueueContent}
        </div>
    )
}

