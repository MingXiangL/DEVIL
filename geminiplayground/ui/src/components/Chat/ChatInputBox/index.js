import ChatInputText from "@/components/Chat/ChatInput";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {Button} from "@/components/ui/button";
import {CornerDownLeft, Mic, Paperclip} from "lucide-react";
import {useRef} from "react";

export default function ChatInputBox({inputTags= [],tagValueAccessor="name", onMessageSend= null}) {
    const inputTextRef = useRef()
    return <form
        onSubmit={(e) => {
            e.preventDefault();
            const inputText = inputTextRef.current;
            let currentValue = inputText.currentValue();
            if (!currentValue) return;
            currentValue =  currentValue.replace(/\[\[(.*?)\]\]/g, (arr => {
                let json = JSON.parse(arr);
                return json[0].map(e => "[" + e[tagValueAccessor] + "]").join(', ');
            }))
            if (onMessageSend) {
                onMessageSend(currentValue);
                inputText.clear();
            }
        }}
        className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-2">
        <ChatInputText ref={inputTextRef} inputTags={inputTags} readOnly={false} />
        <div className="flex items-center p-3 pt-0 mt-2">
            <Button type="submit" size="sm" className="ml-auto gap-1.5">
                Send Message
                <CornerDownLeft className="size-3.5"/>
            </Button>
        </div>
    </form>;
}