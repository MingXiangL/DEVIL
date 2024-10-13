import {useQuery} from "@tanstack/react-query";
import {axiosInstance} from "@/app/axios";
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from "@/components/ui/select";
import {BotIcon} from "lucide-react";
import {FormControl} from "@/components/ui/form";

function beautifyNumber(number) {
    // Check if the input is a number
    if (typeof number !== 'number') {
        return "Error: Input is not a number.";
    }

    // Determine if the number is an integer or a float
    if (Number.isInteger(number)) {
        // For integers, format with thousands separators
        return number.toLocaleString();
    } else {
        // For floats, format with thousands separators and two decimal places
        return number.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }
}

export default function ModelsSelect({...props}) {
    const {data: modelsData, isLoading, refetch} = useQuery({
        queryKey: ['models'],
        queryFn: async () => {
            const {data} = await axiosInstance.get('/models');
            return data;
        },
        select: data => data.filter(({supported_generation_methods}) => supported_generation_methods.includes("generateContent"))
    });


    return (<Select  {...props} >
        <FormControl>
            <SelectTrigger
                className="flex items-stretch [&_[data-description]]:hidden [&_[data-displayname]]:group-hover:flex min-w-[300px]">
                <SelectValue placeholder="Select a model"/>
            </SelectTrigger>
        </FormControl>
        <SelectContent>
            {modelsData?.map(({name, display_name, description, input_token_limit, output_token_limit}) => (
                <SelectItem key={name} value={name}>
                    <div className="flex items-start gap-3 text-muted-foreground w-[300px]">
                        <BotIcon/>
                        <div className="flow flow-col gap-0.5">
                            <p>
                                {name}
                            </p>
                            <p className="text-xs" data-displayname>
                                <span className="font-medium text-foreground">{display_name}</span>
                            </p>
                            <p className="text-xs" data-description>
                                {description}
                            </p>
                            <p className="text-xs" data-inputtokens>
                                <span className="font-medium text-foreground">Input tokens:</span>{' '}
                                <span className="text-muted-foreground">{beautifyNumber(input_token_limit)}</span>
                            </p>
                            <p className="text-xs" data-outputtokens>
                                <span className="font-medium text-foreground">Output tokens:</span>{' '}
                                <span className="text-muted-foreground">{beautifyNumber(output_token_limit)}</span>
                            </p>
                        </div>
                    </div>
                </SelectItem>
            ))}
        </SelectContent>
    </Select>);
}
